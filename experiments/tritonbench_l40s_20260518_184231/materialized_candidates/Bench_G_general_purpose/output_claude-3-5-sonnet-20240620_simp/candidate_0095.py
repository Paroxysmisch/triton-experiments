import torch
import triton
import triton.language as tl

@triton.jit
def fused_recurrent_retention_forward_kernel(
    # Pointers to tensors
    q_ptr, k_ptr, v_ptr, o_ptr, h_ptr,
    initial_state_ptr, final_state_ptr,
    # Dimensions
    B, H, T, D,
    BK: tl.constexpr, BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    BLOCK_SIZE: tl.constexpr):
    
    # Compute indices
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE)
    num_pid_h = H
    num_pid_t = 1
    
    pid_b = pid // (num_pid_h * num_pid_t)
    pid_h = (pid % (num_pid_h * num_pid_t)) // num_pid_t
    
    # Initialize h state
    h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        offs_init = pid_b * stride_qb + pid_h * stride_qh
        h = tl.load(initial_state_ptr + offs_init)
    
    # Block pointers
    offs_q = pid_b * stride_qb + pid_h * stride_qh
    offs_k = pid_b * stride_kb + pid_h * stride_kh
    offs_v = pid_b * stride_vb + pid_h * stride_vh
    offs_o = pid_b * stride_ob + pid_h * stride_oh
    
    # Loop over sequence length
    for t in range(T):
        # Load q, k, v blocks
        q = tl.load(q_ptr + offs_q + t * stride_qt)
        k = tl.load(k_ptr + offs_k + t * stride_kt)
        v = tl.load(v_ptr + offs_v + t * stride_vt)
        
        # Update h state
        h = h + tl.dot(k, v)
        
        # Compute output
        o = tl.dot(q, h)
        
        # Store output
        tl.store(o_ptr + offs_o + t * stride_ot, o)
    
    # Store final state if needed
    if STORE_FINAL_STATE:
        offs_final = pid_b * stride_qb + pid_h * stride_qh
        tl.store(final_state_ptr + offs_final, h)

@triton.jit
def fused_recurrent_retention_backward_kernel(
    # Pointers to tensors
    dq_ptr, dk_ptr, dv_ptr, do_ptr,
    q_ptr, k_ptr, v_ptr, h_ptr,
    # Dimensions and strides similar to forward kernel
    B, H, T, D,
    BK: tl.constexpr, BV: tl.constexpr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE)
    num_pid_h = H
    
    pid_b = pid // num_pid_h
    pid_h = pid % num_pid_h
    
    # Initialize gradients
    dh = tl.zeros([BK, BV], dtype=tl.float32)
    
    # Block pointers
    offs_q = pid_b * stride_qb + pid_h * stride_qh
    offs_k = pid_b * stride_kb + pid_h * stride_kh
    offs_v = pid_b * stride_vb + pid_h * stride_vh
    
    # Backward pass through time
    for t in range(T-1, -1, -1):
        # Load gradients and inputs
        do = tl.load(do_ptr + offs_q + t * stride_qt)
        q = tl.load(q_ptr + offs_q + t * stride_qt)
        k = tl.load(k_ptr + offs_k + t * stride_kt)
        v = tl.load(v_ptr + offs_v + t * stride_vt)
        
        # Compute gradients
        dq = tl.dot(do, dh)
        dk = tl.dot(q, do)
        dv = tl.dot(do, k)
        
        # Update dh
        dh = dh + tl.dot(dk, v) + tl.dot(k, dv)
        
        # Store gradients
        tl.store(dq_ptr + offs_q + t * stride_qt, dq)
        tl.store(dk_ptr + offs_k + t * stride_kt, dk)
        tl.store(dv_ptr + offs_v + t * stride_vt, dv)

class FusedRecurrentRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_state=None):
        B, H, T, D = q.shape
        device = q.device
        
        # Allocate output tensor
        o = torch.empty_like(q)
        final_state = torch.empty((B, H, D, D), device=device) if initial_state is not None else None
        
        # Launch kernel
        grid = (B * H,)
        fused_recurrent_retention_forward_kernel[grid](
            q, k, v, o, None,
            initial_state if initial_state is not None else None,
            final_state if final_state is not None else None,
            B, H, T, D,
            D, D,  # BK, BV
            initial_state is not None,
            final_state is not None,
            *q.stride(), *k.stride(), *v.stride(), *o.stride(),
            BLOCK_SIZE=32
        )
        
        ctx.save_for_backward(q, k, v, o)
        return o, final_state

    @staticmethod
    def backward(ctx, do, dfinal_state=None):
        q, k, v, o = ctx.saved_tensors
        B, H, T, D = q.shape
        
        # Allocate gradient tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # Launch backward kernel
        grid = (B * H,)
        fused_recurrent_retention_backward_kernel[grid](
            dq, dk, dv, do,
            q, k, v, None,
            B, H, T, D,
            D, D,  # BK, BV
            *q.stride(), *k.stride(), *v.stride(),
            BLOCK_SIZE=32
        )
        
        return dq, dk, dv, None

def fused_recurrent_retention(q, k, v, initial_state=None):
    """
    Fused recurrent retention operation.
    
    Args:
        q: Query tensor of shape (B, H, T, D)
        k: Key tensor of shape (B, H, T, D)
        v: Value tensor of shape (B, H, T, D)
        initial_state: Optional initial state tensor of shape (B, H, D, D)
    
    Returns:
        o: Output tensor of shape (B, H, T, D)
        final_state: Optional final state tensor of shape (B, H, D, D)
    """
    return FusedRecurrentRetentionFunction.apply(q, k, v, initial_state)
