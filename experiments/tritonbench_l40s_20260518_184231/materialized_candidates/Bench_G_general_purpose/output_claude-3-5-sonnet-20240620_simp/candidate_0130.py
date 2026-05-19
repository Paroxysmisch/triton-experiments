import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=8),
    ],
    key=['M', 'N']
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k_ptr, v_ptr, d_ptr, v_new_ptr, h_ptr,
    initial_state_ptr, final_state_ptr,
    stride_k_b, stride_k_h, stride_k_t,
    stride_v_b, stride_v_h, stride_v_t,
    stride_d_b, stride_d_h, stride_d_t,
    stride_v_new_b, stride_v_new_h, stride_v_new_t,
    stride_h_b, stride_h_h, stride_h_t,
    B, H, T, D,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr
):
    # Program ID
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # Create block pointers
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr, shape=(B, H, T, D),
        strides=(stride_k_b, stride_k_h, stride_k_t, 1),
        offsets=(pid_b, pid_h, 0, 0),
        block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0, 2, 3)
    )
    
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr, shape=(B, H, T, D),
        strides=(stride_v_b, stride_v_h, stride_v_t, 1),
        offsets=(pid_b, pid_h, 0, 0),
        block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0, 2, 3)
    )
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    # Load initial state if needed
    if USE_INITIAL_STATE:
        acc = tl.load(initial_state_ptr + pid_b * stride_h_b + pid_h * stride_h_h)
    
    # Main loop over time dimension
    for t in range(0, T, BLOCK_SIZE_M):
        # Load k and v blocks
        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)
        
        # Load delta values
        d = tl.load(d_ptr + pid_b * stride_d_b + pid_h * stride_d_h + t * stride_d_t)
        
        # Compute dot product
        dot = tl.dot(k, v)
        
        # Update accumulator
        acc += dot * d
        
        # Store intermediate results in h
        tl.store(h_ptr + pid_b * stride_h_b + pid_h * stride_h_h + t * stride_h_t, acc)
        
        # Update v_new
        v_new = v + acc
        tl.store(v_new_ptr + pid_b * stride_v_new_b + pid_h * stride_v_new_h + t * stride_v_new_t, v_new)
        
        # Advance block pointers
        k_block_ptr = tl.advance(k_block_ptr, (0, 0, BLOCK_SIZE_M, 0))
        v_block_ptr = tl.advance(v_block_ptr, (0, 0, BLOCK_SIZE_M, 0))
    
    # Store final state if needed
    if STORE_FINAL_STATE:
        tl.store(final_state_ptr + pid_b * stride_h_b + pid_h * stride_h_h, acc)

def chunk_fwd_h_fn(k, v, d, initial_state=None, store_final_state=False):
    # Get input dimensions
    B, H, T, D = k.shape
    
    # Create output tensors
    v_new = torch.empty_like(v)
    h = torch.empty((B, H, T, D), device=k.device, dtype=k.dtype)
    final_state = torch.empty((B, H, D), device=k.device, dtype=k.dtype) if store_final_state else None
    
    # Calculate grid size
    grid = (B, H)
    
    # Get strides
    stride_k_b, stride_k_h, stride_k_t = k.stride(0), k.stride(1), k.stride(2)
    stride_v_b, stride_v_h, stride_v_t = v.stride(0), v.stride(1), v.stride(2)
    stride_d_b, stride_d_h, stride_d_t = d.stride(0), d.stride(1), d.stride(2)
    stride_v_new_b, stride_v_new_h, stride_v_new_t = v_new.stride(0), v_new.stride(1), v_new.stride(2)
    stride_h_b, stride_h_h, stride_h_t = h.stride(0), h.stride(1), h.stride(2)
    
    # Launch kernel
    chunk_delta_rule_fwd_kernel_h[grid](
        k, v, d, v_new, h,
        initial_state if initial_state is not None else k.new_zeros(0),
        final_state if final_state is not None else k.new_zeros(0),
        stride_k_b, stride_k_h, stride_k_t,
        stride_v_b, stride_v_h, stride_v_t,
        stride_d_b, stride_d_h, stride_d_t,
        stride_v_new_b, stride_v_new_h, stride_v_new_t,
        stride_h_b, stride_h_h, stride_h_t,
        B, H, T, D,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=store_final_state
    )
    
    return v_new, h, final_state
