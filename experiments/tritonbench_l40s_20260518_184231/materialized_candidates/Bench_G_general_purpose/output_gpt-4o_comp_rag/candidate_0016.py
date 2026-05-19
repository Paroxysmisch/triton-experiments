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
def chunk_global_reversed_cumsum_scalar_kernel(
    s, o,
    s_s_h, s_s_t,
    T: tl.constexpr,
    BT: tl.constexpr
):
    i_bh = tl.program_id(0)
    o_i = tl.arange(0, BT)

    b_z = tl.zeros([BT], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T,), (s_s_t,), (i_t * BT,), (BT,), (1,))
        p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T,), (s_s_t,), (i_t * BT,), (BT,), (1,))
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_c = b_z + b_s
        tl.store(p_o, b_c.to(p_o.dtype.element_ty), boundary_check=(0,))
        b_z += b_s

def chunk_global_reversed_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T = s.shape
    BT = 32

    dtype = dtype or s.dtype
    grid = (B * H,)
    o = torch.empty_like(s, dtype=dtype)
    chunk_global_reversed_cumsum_scalar_kernel[grid](
        s, o,
        s.stride(1), s.stride(2),
        T=T, BT=BT
    )
    return o

# Example usage:
s = torch.randn(2, 3, 4, device='cuda', dtype=torch.float32)
o = chunk_global_reversed_cumsum_scalar(s)
print(o)
