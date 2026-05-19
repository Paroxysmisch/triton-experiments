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
def chunk_global_reversed_cumsum_scalar_kernel(
    s,
    z,
    s_s_h,
    s_s_t,
    T: tl.constexpr,
    BT: tl.constexpr
):
    i_bh = tl.program_id(0)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BT], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T,), (s_s_t,), (i_t * BT,), (BT,), (1,))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T,), (s_s_t,), (i_t * BT,), (BT,), (1,))
        # [BT]
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), boundary_check=(0,))

        if i_t >= 0:
            b_z += tl.sum(b_s, 0)

def chunk_global_reversed_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T = s.shape
    BT = 32  # Block size for the last dimension

    dtype = dtype or s.dtype
    grid = (B * H,)
    z = torch.empty_like(s, dtype=dtype)
    chunk_global_reversed_cumsum_scalar_kernel[grid](
        s, z,
        s.stride(1), s.stride(2),
        T=T, BT=BT
    )
    return z
