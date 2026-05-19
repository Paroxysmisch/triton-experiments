import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_cumsum_scalar_kernel(
    o,
    BT: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    dtype: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_b = i_bh // H
    i_h = i_bh % H

    p_s = tl.make_block_ptr(
        base=o,
        shape=(B, H, T),
        strides=(H * T, T, 1),
        offsets=(i_bh * BT, 0, 0),
        block_shape=(BT, 1),
        order=(1, 0),
    )
    p_o = tl.make_block_ptr(
        base=o,
        shape=(B, H, T),
        strides=(H * T, T, 1),
        offsets=(i_bh * BT, 0, 0),
        block_shape=(BT, 1),
        order=(1, 0),
    )

    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0) + tl.sum(b_s, axis=0)
    tl.store(p_o, b_o.to(dtype), boundary_check=(0, 1))

def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    B, H, T = s.shape
    dtype = dtype or s.dtype
    BT = 128
    o = torch.empty((B, H, T), dtype=dtype, device=s.device)
    chunk_global_cumsum_scalar_kernel[(B * H) // BT](
        o,
        BT,
        B,
        H,
        T,
        dtype=dtype,
    )
    return o
