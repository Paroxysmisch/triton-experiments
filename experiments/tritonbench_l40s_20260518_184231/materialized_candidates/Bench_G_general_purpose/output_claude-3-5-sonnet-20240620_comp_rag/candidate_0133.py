import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k_ptr, v_ptr, d_ptr, v_new_ptr, initial_state_ptr, final_state_ptr,
    stride_kbt, stride_kbk, stride_vbt, stride_vbv, stride_dbt, stride_dbv,
    stride_v_newbt, stride_v_newbv, stride_initial_statebh, stride_final_statebh,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    NT: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    # Create block pointers
    k_block_ptr = tl.make_block_ptr(k_ptr, shape=(BT, BK), strides=(stride_kbt, stride_kbk),
                                    offsets=(0, i_k * BK), block_shape=(BT, BK), order=(1, 0))
    v_block_ptr = tl.make_block_ptr(v_ptr, shape=(BT, BV), strides=(stride_vbt, stride_vbv),
                                    offsets=(0, i_v * BV), block_shape=(BT, BV), order=(1, 0))
    d_block_ptr = tl.make_block_ptr(d_ptr, shape=(BT, BV), strides=(stride_dbt, stride_dbv),
                                    offsets=(0, i_v * BV), block_shape=(BT, BV), order=(1, 0))
    v_new_block_ptr = tl.make_block_ptr(v_new_ptr, shape=(BT, BV), strides=(stride_v_newbt, stride_v_newbv),
                                        offsets=(0, i_v * BV), block_shape=(BT, BV), order=(1, 0))

    # Load initial state if needed
    if USE_INITIAL_STATE:
        b_h = tl.load(initial_state_ptr + i_bh * stride_initial_statebh)
    else:
        b_h = tl.zeros((BK,), dtype=tl.float32)

    # Main loop over time dimension
    for t in range(0, NT, BT):
        # Load k and v
        b_k = tl.load(k_block_ptr)
        b_v = tl.load(v_block_ptr)
        b_d = tl.load(d_block_ptr)

        # Compute h and v_new
        b_h_new = tl.dot(b_k, b_d, allow_tf32=False)
        b_h = b_h + b_h_new
        b_v_new = b_v + tl.dot(b_k.T, b_h, allow_tf32=False)

        # Store v_new
        tl.store(v_new_block_ptr, b_v_new)

        # Update block pointers
        k_block_ptr = tl.advance(k_block_ptr, (BT, 0))
        v_block_ptr = tl.advance(v_block_ptr, (BT, 0))
        d_block_ptr = tl.advance(d_block_ptr, (BT, 0))
        v_new_block_ptr = tl.advance(v_new_block_ptr, (BT, 0))

    # Store final state if needed
    if STORE_FINAL_STATE:
        tl.store(final_state_ptr + i_bh * stride_final_statebh, b_h)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit
def chunk_fwd_h_fn(
    k, v, d, v_new, initial_state, final_state,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    NT: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
):
    # Compute problem size
    grid = (triton.cdiv(K, BK), triton.cdiv(V, BV), 1)

    # Compute strides
    stride_kbt, stride_kbk = k.stride(1), k.stride(2)
    stride_vbt, stride_vbv = v.stride(1), v.stride(2)
    stride_dbt, stride_dbv = d.stride(1), d.stride(2)
    stride_v_newbt, stride_v_newbv = v_new.stride(1), v_new.stride(2)
    stride_initial_statebh = initial_state.stride(0) if USE_INITIAL_STATE else 0
    stride_final_statebh = final_state.stride(0) if STORE_FINAL_STATE else 0

    # Launch kernel
    chunk_delta_rule_fwd_kernel_h[grid](
        k, v, d, v_new, initial_state, final_state,
        stride_kbt, stride_kbk, stride_vbt, stride_vbv, stride_dbt, stride_dbv,
        stride_v_newbt, stride_v_newbv, stride_initial_statebh, stride_final_statebh,
        BT=BT, BK=BK, BV=BV, NT=NT, K=K, V=V,
        USE_INITIAL_STATE=USE_INITIAL_STATE, STORE_FINAL_STATE=STORE_FINAL_STATE,
    )

# Wrapper function to set up and launch the kernel
def chunk_delta_rule_fwd(k, v, d, initial_state=None, store_final_state=False):
    # Extract dimensions
    B, T, K = k.shape
    _, _, V = v.shape
    
    # Compute block sizes
    BT = min(T, 64)
    BK = min(K, 32)
    BV = min(V, 32)
    
    # Initialize output tensors
    v_new = torch.empty_like(v)
    final_state = torch.empty((B, K), dtype=k.dtype, device=k.device) if store_final_state else None
    
    # Launch kernel
    chunk_fwd_h_fn(
        k, v, d, v_new, 
        initial_state if initial_state is not None else torch.empty(0, device=k.device),
        final_state if final_state is not None else torch.empty(0, device=k.device),
        BT=BT, BK=BK, BV=BV, NT=T, K=K, V=V,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=store_final_state,
    )
    
    return v_new, final_state
