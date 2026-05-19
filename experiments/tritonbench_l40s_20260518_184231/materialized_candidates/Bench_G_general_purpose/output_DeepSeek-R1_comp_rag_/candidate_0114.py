import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def chunk_global_cumsum_scalar_kernel(
    s_ptr,
    o_ptr,
    s_s_b,
    s_s_h,
    s_s_t,
    H: tl.constexpr,
    T: tl.constexpr,
    BT: tl.constexpr,
):
    i_bh = tl.program_id(0)
    h = i_bh % H
    b = i_bh // H

    running_sum = tl.zeros((), dtype=tl.float32)
    num_chunks = tl.cdiv(T, BT)

    for i_t in range(num_chunks):
        offset_t = i_t * BT

        block_ptr_s = tl.make_block_ptr(
            base=s_ptr + b * s_s_b + h * s_s_h,
            shape=(T,),
            strides=(s_s_t,),
            offsets=(offset_t,),
            block_shape=(BT,),
            order=(1,)
        )

        block_ptr_o = tl.make_block_ptr(
            base=o_ptr + b * s_s_b + h * s_s_h,
            shape=(T,),
            strides=(s_s_t,),
            offsets=(offset_t,),
            block_shape=(BT,),
            order=(1,)
        )

        current_chunk = tl.load(block_ptr_s, boundary_check=(0,)).to(tl.float32)
        cumsum_chunk = tl.cumsum(current_chunk, axis=0)
        output_chunk = cumsum_chunk + running_sum
        tl.store(block_ptr_o, output_chunk.to(s_ptr.dtype.element_ty), boundary_check=(0,))

        chunk_sum = tl.sum(current_chunk, axis=0)
        running_sum += chunk_sum

def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T = s.shape
    dtype = dtype or s.dtype
    z = torch.empty_like(s, dtype=dtype)

    BT = 512  # Tune this based on your specific hardware and problem size
    grid = (B * H, )

    s_s_b = s.stride(0)
    s_s_h = s.stride(1)
    s_s_t = s.stride(2)

    chunk_global_cumsum_scalar_kernel[grid](
        s, z,
        s_s_b, s_s_h, s_s_t,
        H=H, T=T, BT=BT
    )
    return z
