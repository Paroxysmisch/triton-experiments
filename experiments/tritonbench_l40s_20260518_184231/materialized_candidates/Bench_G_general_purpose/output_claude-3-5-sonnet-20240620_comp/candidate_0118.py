import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, w_ptr, u_ptr, o_ptr,
    initial_state_ptr, final_state_ptr,
    # Matrix dimensions
    T, B, H, D,
    # Parameters
    scale: tl.float32,
    USE_INITIAL_STATE: tl.int32,
    STORE_FINAL_STATE: tl.int32,
    REVERSE: tl.int32,
    # Strides
    stride_t, stride_b, stride_h, stride_d,
    # Block sizes
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_K)
    num_pid_h = tl.cdiv(H, BLOCK_V)
    
    # Block indices
    bid_b = pid // num_pid_h
    bid_h = pid % num_pid_h
    
    # Initialize offsets
    b_start = bid_b * BLOCK_K
    h_start = bid_h * BLOCK_V
    
    # Initialize state
    state = tl.zeros([BLOCK_V], dtype=tl.float32)
    
    # Load initial state if needed
    if USE_INITIAL_STATE:
        offs_init = b_start + tl.arange(0, BLOCK_K)
        mask_init = offs_init < B
        state = tl.load(initial_state_ptr + offs_init, mask=mask_init, other=0.0)
    
    # Time iteration
    for t in range(T):
        t_idx = (T - 1 - t) if REVERSE else t
        
        # Load inputs
        offs_k = t_idx * stride_t + b_start * stride_b + h_start * stride_h + tl.arange(0, BLOCK_V) * stride_d
        offs_v = t_idx * stride_t + b_start * stride_b + h_start * stride_h + tl.arange(0, BLOCK_V) * stride_d
        
        k = tl.load(k_ptr + offs_k)
        v = tl.load(v_ptr + offs_v)
        w = tl.load(w_ptr + offs_k)
        u = tl.load(u_ptr + offs_k)
        
        # Compute recurrent update
        state = state * tl.exp(w) + k * v
        
        # Compute output
        q = tl.load(q_ptr + offs_k)
        output = q * (u + state) * scale
        
        # Store output
        tl.store(o_ptr + offs_k, output)
    
    # Store final state if needed
    if STORE_FINAL_STATE:
        offs_final = b_start + tl.arange(0, BLOCK_K)
        mask_final = offs_final < B
        tl.store(final_state_ptr + offs_final, state, mask=mask_final)

class FusedRecurrentRWKV6Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, w, u, scale=1.0, initial_state=None, store_final_state=False, reverse=False):
        # Save inputs for backward
        ctx.save_for_backward(q, k, v, w, u, initial_state)
        ctx.scale = scale
        ctx.reverse = reverse
        
        # Get dimensions
        T, B, H, D = q.shape
        
        # Compute grid and block sizes
        BK = 32
        BV = 32
        NK = triton.cdiv(B, BK)
        NV = triton.cdiv(H, BV)
        grid = (NK * NV,)
        
        # Allocate output
        output = torch.empty_like(q)
        final_state = torch.empty((B, H, D), device=q.device, dtype=q.dtype) if store_final_state else None
        
        # Launch kernel
        fused_recurrent_rwkv6_fwd_kernel[grid](
            q, k, v, w, u, output,
            initial_state if initial_state is not None else torch.empty(0, device=q.device),
            final_state if final_state is not None else torch.empty(0, device=q.device),
            T, B, H, D,
            scale,
            initial_state is not None,
            store_final_state,
            reverse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            BLOCK_K=BK,
            BLOCK_V=BV,
        )
        
        return output if not store_final_state else (output, final_state)

def fused_recurrent_rwkv6(q, k, v, w, u, scale=1.0, initial_state=None, store_final_state=False, reverse=False):
    """
    Fused recurrent operation for RWKV-6.
    
    Args:
        q, k, v, w, u: Input tensors of shape (T, B, H, D)
        scale: Scaling factor for output
        initial_state: Optional initial state of shape (B, H, D)
        store_final_state: Whether to return the final state
        reverse: Whether to process the sequence in reverse
    
    Returns:
        output: Output tensor of shape (T, B, H, D)
        final_state: Optional final state of shape (B, H, D)
    """
    return FusedRecurrentRWKV6Function.apply(q, k, v, w, u, scale, initial_state, store_final_state, reverse)
