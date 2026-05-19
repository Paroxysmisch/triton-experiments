import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=16),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=32),
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    # Pointers to tensors
    k_ptr, v_ptr, d_ptr, v_new_ptr,
    initial_state_ptr, final_state_ptr,
    # Dimensions
    NT, K, V,
    # Strides
    stride_k_t, stride_k_h, stride_k_k,
    stride_v_t, stride_v_h, stride_v_v,
    stride_d_t, stride_d_h, stride_d_v,
    stride_v_new_t, stride_v_new_h, stride_v_new_v,
    # Options
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    
    # Create block pointers
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(NT, K),
        strides=(stride_k_t, stride_k_k),
        offsets=(0, i_k * BLOCK_SIZE_M),
        block_shape=(1, BLOCK_SIZE_M),
        order=(1, 0)
    )
    
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(NT, V),
        strides=(stride_v_t, stride_v_v),
        offsets=(0, i_v * BLOCK_SIZE_N),
        block_shape=(1, BLOCK_SIZE_N),
        order=(1, 0)
    )
    
    # Initialize hidden state
    b_h = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h = tl.load(initial_state_ptr + i_bh * K + i_k * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    
    # Main loop over time steps
    for t in range(NT):
        # Load k and v blocks
        k = tl.load(k_block_ptr + t * stride_k_t)
        v = tl.load(v_block_ptr + t * stride_v_t)
        
        # Load d block
        d = tl.load(d_ptr + t * stride_d_t + i_bh * stride_d_h + i_v * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        
        # Compute matrix multiplication
        b_h_cumsum = tl.dot(k, d, allow_tf32=False)
        b_h += b_h_cumsum
        
        # Store result
        v_new = v * b_h[:, None]
        tl.store(v_new_ptr + t * stride_v_new_t + i_bh * stride_v_new_h + i_v * BLOCK_SIZE_N,
                v_new)
    
    # Store final state if needed
    if STORE_FINAL_STATE:
        tl.store(final_state_ptr + i_bh * K + i_k * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M),
                b_h)

def chunk_fwd_h_fn(k, v, d, initial_state=None, store_final_state=False):
    # Get dimensions
    BT, BH, BK = k.shape
    _, _, BV = v.shape
    
    # Compute output shapes
    v_new = torch.empty_like(v)
    h = torch.empty((BH, BK), device=k.device, dtype=k.dtype) if store_final_state else None
    
    # Compute grid size
    grid = (
        triton.cdiv(BK, 128),  # Number of blocks for K dimension
        triton.cdiv(BV, 128),  # Number of blocks for V dimension
        BH,                    # Number of batch heads
    )
    
    # Launch kernel
    chunk_delta_rule_fwd_kernel_h[grid](
        k, v, d, v_new,
        initial_state if initial_state is not None else torch.empty(0),
        h if store_final_state else torch.empty(0),
        BT, BK, BV,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        d.stride(0), d.stride(1), d.stride(2),
        v_new.stride(0), v_new.stride(1), v_new.stride(2),
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=store_final_state,
    )
    
    return v_new, h
