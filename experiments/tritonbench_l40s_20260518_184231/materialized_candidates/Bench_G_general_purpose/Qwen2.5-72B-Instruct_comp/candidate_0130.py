import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_V': 64, 'WARPS': 1}, num_warps=1),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_V': 64, 'WARPS': 2}, num_warps=2),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_V': 64, 'WARPS': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_V': 64, 'WARPS': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_V': 64, 'WARPS': 16}, num_warps=16),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_V': 64, 'WARPS': 32}, num_warps=32),
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k_ptr, v_ptr, d_ptr, v_new_ptr, initial_state_ptr, final_state_ptr,
    BT, BK, BV, NT, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_V: tl.constexpr, WARPS: tl.constexpr,
    stride_k_b, stride_k_t, stride_k_k,
    stride_v_b, stride_v_t, stride_v_v,
    stride_d_b, stride_d_t, stride_d_k,
    stride_v_new_b, stride_v_new_t, stride_v_new_v,
    stride_initial_state_b, stride_initial_state_v,
    stride_final_state_b, stride_final_state_v,
):
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)

    # Block pointers
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr, shape=(BT, BK), strides=(stride_k_b, stride_k_k),
        offsets=(i_k * BLOCK_SIZE_K, 0), block_shape=(BLOCK_SIZE_K, 1), order=(1, 0)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr, shape=(BT, BV), strides=(stride_v_b, stride_v_v),
        offsets=(i_v * BLOCK_SIZE_V, 0), block_shape=(BLOCK_SIZE_V, 1), order=(1, 0)
    )
    d_block_ptr = tl.make_block_ptr(
        base=d_ptr, shape=(BT, BK), strides=(stride_d_b, stride_d_k),
        offsets=(i_k * BLOCK_SIZE_K, 0), block_shape=(BLOCK_SIZE_K, 1), order=(1, 0)
    )
    v_new_block_ptr = tl.make_block_ptr(
        base=v_new_ptr, shape=(BT, BV), strides=(stride_v_new_b, stride_v_new_v),
        offsets=(i_v * BLOCK_SIZE_V, 0), block_shape=(BLOCK_SIZE_V, 1), order=(1, 0)
    )
    h_block_ptr = tl.make_block_ptr(
        base=tl.empty((1, 1), tl.float32), shape=(1, 1), strides=(1, 1),
        offsets=(0, 0), block_shape=(1, 1), order=(1, 0)
    )

    if USE_INITIAL_STATE:
        initial_state_block_ptr = tl.make_block_ptr(
            base=initial_state_ptr, shape=(BT, BV), strides=(stride_initial_state_b, stride_initial_state_v),
            offsets=(i_v * BLOCK_SIZE_V, 0), block_shape=(BLOCK_SIZE_V, 1), order=(1, 0)
        )
        h = tl.load(initial_state_block_ptr)
    else:
        h = tl.zeros((BLOCK_SIZE_V,), dtype=tl.float32)

    b_h_cumsum = tl.zeros((BLOCK_SIZE_V,), dtype=tl.float32)

    for t in range(NT):
        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)
        d = tl.load(d_block_ptr)

        # Compute the new state
        h = h + tl.dot(k, d, allow_tf32=False)

        # Update v_new
        v_new = v + h
        tl.store(v_new_block_ptr, v_new)

        # Update cumulative state
        b_h_cumsum += h

        # Move to the next time step
        k_block_ptr = tl.advance(k_block_ptr, (1, 0))
        v_block_ptr = tl.advance(v_block_ptr, (1, 0))
        d_block_ptr = tl.advance(d_block_ptr, (1, 0))
        v_new_block_ptr = tl.advance(v_new_block_ptr, (1, 0))

    if STORE_FINAL_STATE:
        final_state_block_ptr = tl.make_block_ptr(
            base=final_state_ptr, shape=(BT, BV), strides=(stride_final_state_b, stride_final_state_v),
            offsets=(i_v * BLOCK_SIZE_V, 0), block_shape=(BLOCK_SIZE_V, 1), order=(1, 0)
        )
        tl.store(final_state_block_ptr, b_h_cumsum)

import torch

def chunk_fwd_h_fn(k, v, d, initial_state=None, final_state=None, use_initial_state=False, store_final_state=False):
    BT, NT, BK = k.shape
    _, _, BV = v.shape
    _, _, _ = d.shape

    # Initialize output tensors
    v_new = torch.empty_like(v)
    h = torch.zeros((BT, BV), device=k.device, dtype=k.dtype)

    # Determine block sizes
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_V = 64
    MAX_WARP_SIZE = 32

    # Compute grid size
    grid = (
        (BK + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K,
        (BV + BLOCK_SIZE_V - 1) // BLOCK_SIZE_V,
        BT
    )

    # Strides
    stride_k_b = k.stride(0)
    stride_k_t = k.stride(1)
    stride_k_k = k.stride(2)
    stride_v_b = v.stride(0)
    stride_v_t = v.stride(1)
    stride_v_v = v.stride(2)
    stride_d_b = d.stride(0)
    stride_d_t = d.stride(1)
    stride_d_k = d.stride(2)
    stride_v_new_b = v_new.stride(0)
    stride_v_new_t = v_new.stride(1)
    stride_v_new_v = v_new.stride(2)
    stride_initial_state_b = initial_state.stride(0) if initial_state is not None else 0
    stride_initial_state_v = initial_state.stride(1) if initial_state is not None else 0
    stride_final_state_b = final_state.stride(0) if final_state is not None else 0
    stride_final_state_v = final_state.stride(1) if final_state is not None else 0

    # Launch the kernel
    chunk_delta_rule_fwd_kernel_h[grid](
        k, v, d, v_new, initial_state, final_state,
        BT, BK, BV, NT, use_initial_state, store_final_state,
        BLOCK_SIZE_K, BLOCK_SIZE_V, MAX_WARP_SIZE,
        stride_k_b, stride_k_t, stride_k_k,
        stride_v_b, stride_v_t, stride_v_v,
        stride_d_b, stride_d_t, stride_d_k,
        stride_v_new_b, stride_v_new_t, stride_v_new_v,
        stride_initial_state_b, stride_initial_state_v,
        stride_final_state_b, stride_final_state_v
    )

    return v_new, h
