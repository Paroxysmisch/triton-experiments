import torch
import triton
import triton.language as tl

# Constants for block sizes and other parameters
BLOCK_M = 128
BLOCK_DMODEL = 64
BLOCK_N = 64

@triton.jit
def fused_recurrent_retention_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, o_ptr, state_ptr,
    # Matrix dimensions
    B, H, T, D,
    # Strides for the different matrices
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    # Other parameters
    scale, USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    num_warps: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute batch and head indices
    b_h = pid // (T)
    t = pid % T
    
    b = b_h // H
    h = b_h % H
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    # Load initial state if needed
    if USE_INITIAL_STATE and t == 0:
        acc = tl.load(state_ptr + b * stride_qb + h * stride_qh)
    
    # Compute decay factor based on head index
    decay = tl.math.exp(-tl.math.log(2.0) * (h + 1))
    
    # Load query block
    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + t * stride_qt + 
                tl.arange(0, BLOCK_DMODEL) * stride_qd)
    
    # Scale query
    q = q * scale
    
    # Main loop over time steps
    for tau in range(t):
        # Load key and value
        k = tl.load(k_ptr + b * stride_kb + h * stride_kh + tau * stride_kt +
                    tl.arange(0, BLOCK_DMODEL) * stride_kd)
        v = tl.load(v_ptr + b * stride_vb + h * stride_vh + tau * stride_vt +
                    tl.arange(0, BLOCK_DMODEL) * stride_vd)
        
        # Compute attention scores
        s = tl.sum(q * k)
        s = tl.math.exp(s)
        
        # Update accumulator
        acc = acc * decay + s * v
    
    # Store output
    tl.store(o_ptr + b * stride_ob + h * stride_oh + t * stride_ot +
             tl.arange(0, BLOCK_DMODEL) * stride_od, acc)
    
    # Store final state if needed
    if STORE_FINAL_STATE and t == T-1:
        tl.store(state_ptr + b * stride_qb + h * stride_qh, acc)

@triton.jit
def fused_recurrent_retention_bwd_kernel(
    # Pointers to matrices
    dq_ptr, dk_ptr, dv_ptr, do_ptr,
    q_ptr, k_ptr, v_ptr,
    # Matrix dimensions and strides (same as forward pass)
    B, H, T, D,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    # Other parameters
    scale,
    num_warps: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute indices
    b_h = pid // T
    t = pid % T
    
    b = b_h // H
    h = b_h % H
    
    # Initialize gradients
    dq = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    # Compute decay factor
    decay = tl.math.exp(-tl.math.log(2.0) * (h + 1))
    
    # Load do
    do = tl.load(do_ptr + b * stride_ob + h * stride_oh + t * stride_ot +
                 tl.arange(0, BLOCK_DMODEL) * stride_od)
    
    # Backward pass through time
    for tau in range(t, T):
        # Load query, key, value
        q = tl.load(q_ptr + b * stride_qb + h * stride_qh + tau * stride_qt +
                    tl.arange(0, BLOCK_DMODEL) * stride_qd)
        k = tl.load(k_ptr + b * stride_kb + h * stride_kh + t * stride_kt +
                    tl.arange(0, BLOCK_DMODEL) * stride_kd)
        v = tl.load(v_ptr + b * stride_vb + h * stride_vh + t * stride_vt +
                    tl.arange(0, BLOCK_DMODEL) * stride_vd)
        
        # Scale query
        q = q * scale
        
        # Compute attention scores
        s = tl.sum(q * k)
        s = tl.math.exp(s)
        
        # Compute gradients
        ds = s * tl.sum(do * v)
        dq += ds * k * scale
        dk += ds * q
        dv += s * do
        
        # Update do with decay
        do = do * decay
    
    # Store gradients
    tl.store(dq_ptr + b * stride_qb + h * stride_qh + t * stride_qt +
             tl.arange(0, BLOCK_DMODEL) * stride_od, dq)
    tl.store(dk_ptr + b * stride_kb + h * stride_kh + t * stride_kt +
             tl.arange(0, BLOCK_DMODEL) * stride_kd, dk)
    tl.store(dv_ptr + b * stride_vb + h * stride_vh + t * stride_vt +
             tl.arange(0, BLOCK_DMODEL) * stride_vd, dv)

def fused_recurrent_retention(q, k, v, initial_state=None, store_final_state=False):
    """
    Fused recurrent retention mechanism implementation.
    
    Args:
        q: Query tensor of shape (B, H, T, D)
        k: Key tensor of shape (B, H, T, D)
        v: Value tensor of shape (B, H, T, D)
        initial_state: Optional initial state tensor of shape (B, H, D)
        store_final_state: Whether to store and return the final state
    
    Returns:
        o: Output tensor of shape (B, H, T, D)
        final_state: Final state tensor of shape (B, H, D) if store_final_state=True
    """
    B, H, T, D = q.shape
    
    # Allocate output tensor
    o = torch.empty_like(q)
    
    # Compute scale factor
    scale = 1.0 / (D ** 0.5)
    
    # Prepare grid
    grid = (B * H * T,)
    
    # Prepare state tensor if needed
    if initial_state is not None or store_final_state:
        state = torch.zeros((B, H, D), device=q.device, dtype=q.dtype)
        if initial_state is not None:
            state.copy_(initial_state)
    else:
        state = None
    
    # Launch forward kernel
    fused_recurrent_retention_fwd_kernel[grid](
        q, k, v, o, state,
        B, H, T, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        scale,
        initial_state is not None,
        store_final_state,
        num_warps=4
    )
    
    if store_final_state:
        return o, state
    return o

class FusedRecurrentRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_state=None, store_final_state=False):
        ctx.save_for_backward(q, k, v)
        return fused_recurrent_retention(q, k, v, initial_state, store_final_state)
    
    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        B, H, T, D = q.shape
        
        # Allocate gradient tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # Compute scale factor
        scale = 1.0 / (D ** 0.5)
        
        # Prepare grid
        grid = (B * H * T,)
        
        # Launch backward kernel
        fused_recurrent_retention_bwd_kernel[grid](
            dq, dk, dv, do,
            q, k, v,
            B, H, T, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            scale,
            num_warps=4
        )
        
        return dq, dk, dv, None, None
