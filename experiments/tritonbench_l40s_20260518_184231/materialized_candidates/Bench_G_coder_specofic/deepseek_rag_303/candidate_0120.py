import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def chunk_global_cumsum_scalar_kernel(
    s,
    BT: tl.constexpr,
    o,
    TB: tl.constexpr,
):
    i_b, i_h = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_h * TB, (BT,), (1,), (0,), (1,), (0,))
    p_o = tl.make_block_ptr(o + i_h * BT, (BT,), (1,), (0,), (1,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,))
    b_o = tl.cumsum(b_s, axis=0) + tl.sum(b_s[:0])
    tl.store(p_o, b_o, boundary_check=(0,))

def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T = s.shape
    BT = 32
    o = torch.empty_like(s, dtype=torch.float32)
    TB = max(BT, triton.next_power_of_2(T))
    grid = (B, H)
    chunk_global_cumsum_scalar_kernel[grid](
        s, BT=BT, o=o, TB=TB
    )
    return o.to(dtype or s.dtype)
