import torch
import triton
import triton.language as tl

@triton.jit
def chunk_retention_fwd_kernel_h(
    k_ptr, v_ptr, h_ptr, 
    initial_state_ptr, final_state_ptr,
    NT, D, 
    stride_k_t, stride_k_d,
    stride_v_t, stride_v_d,
    stride_h_t, stride_h_d,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Initialize buffer for h
    b_h = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load initial state if needed
    if USE_INITIAL_STATE:
        b_h = tl.load(initial_state_ptr + pid * stride_h_d)
    
    # Compute decay factors
    def compute_decay(t):
        return tl.exp(-t / 16.0)  # Example decay function
    
    # Main loop over time dimension
    for t in range(0, NT, 1):
        # Load k and v blocks
        k = tl.load(k_ptr + t * stride_k_t + pid * stride_k_d)
        v = tl.load(v_ptr + t * stride_v_t + pid * stride_v_d)
        
        # Compute decay factors
        d_b = compute_decay(tl.float32(NT - t - 1))
        d_i = compute_decay(tl.float32(1.0))
        
        # Update h
        b_h = d_i * b_h + k * v
        
        # Store h
        tl.store(h_ptr + t * stride_h_t + pid * stride_h_d, b_h)
    
    # Store final state if needed
    if STORE_FINAL_STATE:
        tl.store(final_state_ptr + pid * stride_h_d, b_h)

@triton.jit
def chunk_retention_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, o_ptr,
    NT, D,
    stride_q_t, stride_q_d,
    stride_k_t, stride_k_d,
    stride_v_t, stride_v_d,
    stride_h_t, stride_h_d,
    stride_o_t, stride_o_d,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Initialize output buffers
    b_o = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    b_s = tl.zeros([1], dtype=tl.float32)
    
    def compute_decay(t):
        return tl.exp(-t / 16.0)
    
    # Main loop over time dimension
    for t in range(0, NT, 1):
        # Load blocks
        q = tl.load(q_ptr + t * stride_q_t + pid * stride_q_d)
        k = tl.load(k_ptr + t * stride_k_t + pid * stride_k_d)
        v = tl.load(v_ptr + t * stride_v_t + pid * stride_v_d)
        h = tl.load(h_ptr + t * stride_h_t + pid * stride_h_d)
        
        # Compute decay factor
        d_i = compute_decay(tl.float32(1.0))
        
        # Update output
        qk = q * k
        b_s += qk
        b_o = b_o + qk * v + q * h
        
        # Store output
        tl.store(o_ptr + t * stride_o_t + pid * stride_o_d, b_o / (b_s + 1e-6))

@triton.jit
def chunk_retention_bwd_kernel_dh(
    do_ptr, q_ptr, dh_ptr,
    NT, D,
    stride_do_t, stride_do_d,
    stride_q_t, stride_q_d,
    stride_dh_t, stride_dh_d,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    b_dh = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Backward loop over time
    for t in range(NT-1, -1, -1):
        do = tl.load(do_ptr + t * stride_do_t + pid * stride_do_d)
        q = tl.load(q_ptr + t * stride_q_t + pid * stride_q_d)
        
        b_dh = b_dh + do * q
        tl.store(dh_ptr + t * stride_dh_t + pid * stride_dh_d, b_dh)

@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    do_ptr, q_ptr, k_ptr, v_ptr, h_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    NT, D,
    stride_do_t, stride_do_d,
    stride_q_t, stride_q_d,
    stride_k_t, stride_k_d,
    stride_v_t, stride_v_d,
    stride_h_t, stride_h_d,
    stride_dq_t, stride_dq_d,
    stride_dk_t, stride_dk_d,
    stride_dv_t, stride_dv_d,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    def compute_decay(t):
        return tl.exp(-t / 16.0)
    
    for t in range(0, NT, 1):
        # Load blocks
        do = tl.load(do_ptr + t * stride_do_t + pid * stride_do_d)
        q = tl.load(q_ptr + t * stride_q_t + pid * stride_q_d)
        k = tl.load(k_ptr + t * stride_k_t + pid * stride_k_d)
        v = tl.load(v_ptr + t * stride_v_t + pid * stride_v_d)
        h = tl.load(h_ptr + t * stride_h_t + pid * stride_h_d)
        
        # Compute gradients
        d_i = compute_decay(tl.float32(1.0))
        
        dq = do * (k * v + h)
        dk = do * q * v
        dv = do * (q * k)
        
        # Store gradients
        tl.store(dq_ptr + t * stride_dq_t + pid * stride_dq_d, dq)
        tl.store(dk_ptr + t * stride_dk_t + pid * stride_dk_d, dk)
        tl.store(dv_ptr + t * stride_dv_t + pid * stride_dv_d, dv)

class ChunkRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_state=None, store_final_state=False):
        NT, B, D = q.shape
        device = q.device
        
        # Allocate output tensors
        h = torch.zeros((NT, B, D), device=device)
        o = torch.zeros((NT, B, D), device=device)
        final_state = torch.zeros((B, D), device=device) if store_final_state else None
        
        # Configure grid and block sizes
        grid = (D,)
        
        # Launch forward kernels
        chunk_retention_fwd_kernel_h[grid](
            k, v, h,
            initial_state if initial_state is not None else torch.zeros_like(h[0]),
            final_state if store_final_state else torch.zeros_like(h[0]),
            NT, D,
            k.stride(0), k.stride(2),
            v.stride(0), v.stride(2),
            h.stride(0), h.stride(2),
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=store_final_state,
            BLOCK_SIZE=32,
        )
        
        chunk_retention_fwd_kernel_o[grid](
            q, k, v, h, o,
            NT, D,
            q.stride(0), q.stride(2),
            k.stride(0), k.stride(2),
            v.stride(0), v.stride(2),
            h.stride(0), h.stride(2),
            o.stride(0), o.stride(2),
            BLOCK_SIZE=32,
        )
        
        ctx.save_for_backward(q, k, v, h)
        return o, final_state if store_final_state else None
    
    @staticmethod
    def backward(ctx, do, dfinal_state=None):
        q, k, v, h = ctx.saved_tensors
        NT, B, D = q.shape
        device = q.device
        
        # Allocate gradient tensors
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dh = torch.zeros_like(h)
        
        # Configure grid
        grid = (D,)
        
        # Launch backward kernels
        chunk_retention_bwd_kernel_dh[grid](
            do, q, dh,
            NT, D,
            do.stride(0), do.stride(2),
            q.stride(0), q.stride(2),
            dh.stride(0), dh.stride(2),
            BLOCK_SIZE=32,
        )
        
        chunk_retention_bwd_kernel_dqkv[grid](
            do, q, k, v, h,
            dq, dk, dv,
            NT, D,
            do.stride(0), do.stride(2),
            q.stride(0), q.stride(2),
            k.stride(0), k.stride(2),
            v.stride(0), v.stride(2),
            h.stride(0), h.stride(2),
            dq.stride(0), dq.stride(2),
            dk.stride(0), dk.stride(2),
            dv.stride(0), dv.stride(2),
            BLOCK_SIZE=32,
        )
        
        return dq, dk, dv, None, None

def chunk_retention(q, k, v, initial_state=None, store_final_state=False):
    """
    Applies chunk retention operation on input tensors.
    
    Args:
        q: Query tensor of shape (NT, B, D)
        k: Key tensor of shape (NT, B, D)
        v: Value tensor of shape (NT, B, D)
        initial_state: Optional initial state tensor of shape (B, D)
        store_final_state: Whether to store and return the final state
    
    Returns:
        output: Output tensor of shape (NT, B, D)
        final_state: Final state tensor of shape (B, D) if store_final_state=True, else None
    """
    return ChunkRetentionFunction.apply(q, k, v, initial_state, store_final_state)
