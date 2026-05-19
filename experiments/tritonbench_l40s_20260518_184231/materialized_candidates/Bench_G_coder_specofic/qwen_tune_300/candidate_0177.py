import torch
import triton
import triton.language as tl
from typing import Optional

@triton.autotune(
    configs=[
        triton.Config({"S": 64}, num_warps=4),
        triton.Config({"S": 128}, num_warps=8),
        triton.Config({"S": 256}, num_warps=16),
        triton.Config({"S": 512}, num_warps=32),
    ],
    key=["BT", "BS"],
)
@triton.jit
def chunk_global_cumsum_vector_kernel(
    s,
    z,
    stride: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    """Chunk global cumsum, vectorized over the last dimension."""
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    m_s = tl.arange(0, BS)[:, None] >= tl.arange(0, BS)[None, :]
    BT = min(BT, T - i_s * BT)

    b_z = tl.zeros([BT, BS], dtype=tl.float32)
    for i_t in range(0, tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * stride, (T, B, H), (BT, BT, BS), (i_s * BT, i_t * BT, i_s * BS), (BT, BT, BS), (1, 0, 0))
        p_z = tl.make_block_ptr(z + i_bh * stride, (T, B, H), (BT, BT, BS), (i_s * BT, i_t * BT, i_s * BS), (BT, BT, BS), (1, 0, 0))
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_c = tl.dot(m_s, b_s)
        b_c = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BS)[None, :], b_c, 0)
        b_z += b_c
        tl.store(p_z, b_z.to(p_z.dtype.element_ty), boundary_check=(0, 1))


def chunk_global_cumsum_vector(s: torch.Tensor, chunk_size: int, ignore_index: Optional[int] = None):
    """Global cumulative sum over blocks."""
    B, H, T, S = s.shape
    z = torch.empty_like(s, dtype=s.dtype)
    BT = chunk_size

    def grid(meta):
        return (triton.cdiv(S, meta["BS"]), B * H)

    chunk_global_cumsum_vector_kernel[grid](s, z, stride=s.stride(1), B=B, H=H, T=T, BT=BT, BS=min(32, triton.next_power_of_2(S)))
    return z
