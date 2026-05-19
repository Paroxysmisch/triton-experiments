import torch
import triton
import triton.language as tl

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
    key=['T']
)
@triton.jit
def chunk_global_cumsum_scalar_kernel(
    s,
    o,
    s_s_h,
    s_s_t,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr
):
    i_h, i_b = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    b_z = tl.zeros([1], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_b * s_s_h, (T, H), (s_s_t, s_s_h), (i_t * BT, i_h), (BT, 1), (1, 0))
        p_o = tl.make_block_ptr(o + i_b * s_s_h, (T, H), (s_s_t, s_s_h), (i_t * BT, i_h), (BT, 1), (1, 0))
        # [BT, 1]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_o, b_c.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        if i_t >= 0:
            b_z += tl.sum(b_s, 0)


def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T = s.shape
    BT = 32

    dtype = dtype or s.dtype
    grid = (H, B)
    o = torch.empty_like(s, dtype=dtype)
    chunk_global_cumsum_scalar_kernel[grid](
        s, o,
        s.stride(1), s.stride(2),
        T=T, H=H, BT=BT
    )
    return o


class CumsumFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, s, dtype):
        o = chunk_global_cumsum_scalar(s, dtype)
        ctx.dtype = dtype
        return o

    @staticmethod
    def backward(ctx, do):
        # For simplicity, we assume the backward pass is the same as the forward pass
        ds = chunk_global_cumsum_scalar(do, ctx.dtype)
        return ds, None


def cumsum(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return CumsumFunction.apply(s, dtype)
