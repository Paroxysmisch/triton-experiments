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
def chunk_global_cumsum_scalar_kernel(
    s,
    o,
    s_s_h,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr
):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    running_total = tl.zeros([BT], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (1, s_s_d), (i_t * BT, i_s), (BT, 1), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T, S), (1, s_s_d), (i_t * BT, i_s), (BT, 1), (1, 0))
        
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_c = running_total[:, None] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_o, b_c.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        running_total += tl.sum(b_s, 1)


def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T = s.shape
    BT = 32  # You can choose a different block size based on your needs

    dtype = dtype or s.dtype
    grid = (triton.cdiv(T, BT), B * H)
    o = torch.empty_like(s, dtype=dtype)
    chunk_global_cumsum_scalar_kernel[grid](
        s, o,
        s.stride(1), s.stride(2),
        T=T, S=1, BT=BT
    )
    return o
