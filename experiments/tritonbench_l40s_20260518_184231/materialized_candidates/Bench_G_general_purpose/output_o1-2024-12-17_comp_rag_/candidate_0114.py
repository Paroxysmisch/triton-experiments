import torch
import triton
import triton.language as tl
from typing import Optional

@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 16}, num_warps=4),
        triton.Config({'BT': 16}, num_warps=8),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=4),
        triton.Config({'BT': 64}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def chunk_cumsum_fwd_kernel(
    s,
    z,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    # Create a mask for cumulative sum within a block
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)

    # Carry-over sum across blocks
    b_z = tl.zeros([BS], dtype=tl.float32)

    # Process the input in chunks along time dimension
    for i_t in range(tl.cdiv(T, BT)):
        ptr_s = tl.make_block_ptr(
            base=s + i_bh * s_s_h,
            shape=(T, S),
            strides=(s_s_t, s_s_d),
            offsets=(i_t * BT, i_s * BS),
            block_shape=(BT, BS),
            order=(1, 0)
        )
        ptr
