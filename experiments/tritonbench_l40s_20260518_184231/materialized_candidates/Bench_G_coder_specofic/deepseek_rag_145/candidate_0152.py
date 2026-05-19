import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.autotune(
    configs=[
        triton.Config({'BD': 32}, num_warps=1),
        triton.Config({'BD': 32}, num_warps=2),
        triton.Config({'BD': 32}, num_warps=4),
        triton.Config({'BD': 32}, num_warps=8),
        triton.Config({'BD': 64}, num_warps=1),
        triton.Config({'BD': 64}, num_warps=2),
        triton.Config({'BD': 64}, num_warps=4),
        triton.Config({'BD': 64}, num_warps=8),
        triton.Config({'BD': 128}, num_warps=1),
        triton.Config({'BD': 128}, num_warps=2),
        triton.Config({'BD': 128}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def chunk_retention_fwd_kernel_h(
    k,
    v,
    h0,
    h,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_k = k + i_bh * T * D + i_t * BT * D + o_d
    p_v = v + i_bh * T * D + i_t * BT * D + o_d
    p_h = h + i_bh * T * D + i_t * BT * D + o_d
    final_state = h + i_bh * D + o_d

    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if i_t == 0:
            b_h += tl.load(h0 + i_bh * D + o_d, mask=mask, other=0).to(tl.float32)

    # Compute decay factors d_b, d_i. (Implement your decay function here)

    for i in range(0, BT):
        mask_t = mask & ((i_t * BT + i) < T)
        b_k = tl.load(p_k, mask=mask_t, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_t, other=0).to(tl.float32)

        b_h = d_b * b_h + d_i * tl.dot(b_k, b_v)
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_t)

        p_k += D
        p_v += D
        p_h += D

    if STORE_FINAL_STATE and i_t == NT - 1:
        tl.store(final_state, b_h.to(final_state.dtype.element_ty))
