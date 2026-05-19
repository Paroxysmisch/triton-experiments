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
    key=['T']
)
@triton.jit
def chunk_global_cumsum_scalar_kernel(
    s,  # Input tensor
    o,  # Output tensor
    s_s_h,  # Stride of the head dimension
    s_s_t,  # Stride of the time dimension
    T: tl.constexpr,  # Time dimension size
    BT: tl.constexpr  # Block size for the time dimension
):
    i_bh = tl.program_id(0)  # Batch and head index
    o_i = tl.arange(0, BT)  # Index range for the block
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)  # Mask for cumulative sum

    b_z = tl.zeros([BT], dtype=tl.float32)  # Running sum
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T,), (s_s_t,), (i_t * BT,), (BT,), (1,))
        p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T,), (s_s_t,), (i_t * BT,), (BT,), (1,))
        
        # Load block and perform cumulative sum
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_c = b_z + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_o, b_c.to(p_o.dtype.element_ty), boundary_check=(0,))

        # Update running sum
        if i_t >= 0:
            b_z += tl.sum(b_s)


def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T = s.shape
    BT = 32  # Block size for the time dimension

    dtype = dtype or s.dtype
    grid = (B * H,)
    o = torch.empty_like(s, dtype=dtype)
    chunk_global_cumsum_scalar_kernel[grid](
        s, o,
        s.stride(1), s.stride(2),
        T=T, BT=BT
    )
    return o
