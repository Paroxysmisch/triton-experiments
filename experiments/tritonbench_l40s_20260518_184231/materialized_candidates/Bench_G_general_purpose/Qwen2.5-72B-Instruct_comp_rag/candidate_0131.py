import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_V': 128, 'WARPS': 1}, num_warps=1),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_V': 128, 'WARPS': 2}, num_warps=2),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_V': 128, 'WARPS': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_V': 128, 'WARPS': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_V': 128, 'WARPS': 16}, num_warps=16),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_V': 128, 'WARPS': 32}, num_warps=32),
    ],
    key=['BT', 'BK', 'BV']
)
@triton.jit
def chunk_delta_rule_fwd_kernel_h(
    k_ptr, k_stride_b, k_stride_t, k_stride_k,
    v_ptr, v_stride_b, v_stride_t, v_stride_v,
    d_ptr, d_stride_b, d_stride_t, d_stride_v,
    v_new_ptr, v_new_stride_b, v_new_stride_t, v_new_stride_v,
    initial_state_ptr, initial_state_stride_b, initial_state_stride_v,
    final_state_ptr, final_state_stride_b, final_state_stride_v,
    NT, BT, BK, BV,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    WARPS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_blocks = (BT * BK * BV) // (BLOCK_SIZE_K * BLOCK_SIZE_V)
    if pid >= num_blocks:
        return

    i_k = pid % BK
    i_v = (pid // BK) % BV
    i_bh = (pid // (BK * BV)) % BT

    b_h = tl.zeros((BLOCK_SIZE_V,), dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h = tl.load(initial_state_ptr + i_bh * initial_state_stride_b + i_v * initial_state_stride_v)

    for t in range(NT):
        k_block = tl.load(k_ptr + i_bh * k_stride_b + t * k_stride_t + i_k * k_stride_k)
        v_block = tl.load(v_ptr + i_bh * v_stride_b + t * v_stride_t + i_v * v_stride_v)
        d_block = tl.load(d_ptr + i_bh * d_stride_b + t * d_stride_t + i_v * d_stride_v)

        b_h += tl.dot(k_block, d_block, allow_tf32=False)
        v_new_block = v_block + b_h
        tl.store(v_new_ptr + i_bh * v_new_stride_b + t * v_new_stride_t + i_v * v_new_stride_v, v_new_block)

    if STORE_FINAL_STATE:
        tl.store(final_state_ptr + i_bh * final_state_stride_b + i_v * final_state_stride_v, b_h)

import torch

def chunk_fwd_h_fn(
    k: torch.Tensor,
    v: torch.Tensor,
    d: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    NT: int = 1,
    BT: int = 1,
    BK: int = 1,
    BV: int = 1,
    USE_INITIAL_STATE: bool = False,
    STORE_FINAL_STATE: bool = False
):
    assert k.is_cuda and v.is_cuda and d.is_cuda, "Input tensors must be on GPU"
    assert k.dtype == v.dtype == d.dtype, "Input tensors must have the same data type"
    assert k.shape == (BT, NT, BK), "Input tensor k must have shape (BT, NT, BK)"
    assert v.shape == (BT, NT, BV), "Input tensor v must have shape (BT, NT, BV)"
    assert d.shape == (BT, NT, BV), "Input tensor d must have shape (BT, NT, BV)"

    if USE_INITIAL_STATE:
        assert initial_state is not None, "Initial state must be provided if USE_INITIAL_STATE is True"
        assert initial_state.shape == (BT, BV), "Initial state must have shape (BT, BV)"
        assert initial_state.dtype == k.dtype, "Initial state must have the same data type as input tensors"

    if STORE_FINAL_STATE:
        assert final_state is not None, "Final state must be provided if STORE_FINAL_STATE is True"
        assert final_state.shape == (BT, BV), "Final state must have shape (BT, BV)"
        assert final_state.dtype == k.dtype, "Final state must have the same data type as input tensors"

    v_new = torch.empty_like(v)

    grid = (BT * BK * BV // (128 * 128),)
    chunk_delta_rule_fwd_kernel_h[grid](
        k, k.stride(0), k.stride(1), k.stride(2),
        v, v.stride(0), v.stride(1), v.stride(2),
        d, d.stride(0), d.stride(1), d.stride(2),
        v_new, v_new.stride(0), v_new.stride(1), v_new.stride(2),
        initial_state, initial_state.stride(0), initial_state.stride(1) if USE_INITIAL_STATE else 0,
        final_state, final_state.stride(0), final_state.stride(1) if STORE_FINAL_STATE else 0,
        NT, BT, BK, BV,
        USE_INITIAL_STATE,
        STORE_FINAL_STATE,
        BLOCK_SIZE_K=128,
        BLOCK_SIZE_V=128,
        WARPS=32
    )

    return v_new
