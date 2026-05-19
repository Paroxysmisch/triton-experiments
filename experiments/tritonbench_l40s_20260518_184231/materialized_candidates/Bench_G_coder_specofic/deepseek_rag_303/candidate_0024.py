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
def chunk_global_reversed_cumsum_scalar_kernel(
    s,
    o,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    i_bh = tl.program_id(0)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
        # [BT, S]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        if i_t != T // BT - 1:
            b_z.to(b_s.dtype)
            tl.store(p_o, b_z.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        else:
            b_z = tl.full([BS], float('inf'), dtype=tl.float32)
            b_z.to(b_s.dtype)
            tl.store(p_o, b_z.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        b_c = b_z[None, :] - tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_o, b_c.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        b_z = tl.sum(b_s, 0)


def chunk_global_reversed_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T, S = s.shape
    BS = 32

    dtype = dtype or s.dtype
    grid = (B * H,)
    o = torch.empty_like(s, dtype=dtype)
    chunk_global_reversed_cumsum_scalar_kernel[grid](
        s, o,
        s.stride(1), s.stride(2), s.stride(3),
        T=T, S=S, BT=16, BS=32
    )
    return o


class GlobalReversedCumsumFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, s, dtype):
        o = chunk_global_reversed_cumsum_scalar(s, dtype)
        ctx.dtype = dtype
        return o

    @staticmethod
    def backward(ctx, do):
        di = chunk_global_reversed_cumsum_scalar(do, ctx.dtype)
        return di, None


def global_reversed_cumsum(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return GlobalReversedCumsumFunction.apply(s, dtype)
