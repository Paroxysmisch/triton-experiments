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
def chunk_global_reversed_cumsum_vector_kernel(
    s_ptr,
    z_ptr,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    i_s = tl.program_id(0)
    i_bh = tl.program_id(1)
    
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    
    b_z = tl.zeros([BS], dtype=tl.float32)
    num_blocks = tl.cdiv(T, BT)
    
    for i_t in range(num_blocks - 1, -1, -1):
        block_start = i_t * BT
        
        s_block_ptr = tl.make_block_ptr(
            base=s_ptr + i_bh * s_s_h,
            shape=(T, S),
            strides=(s_s_t, s_s_d),
            offsets=(block_start, i_s * BS),
            block_shape=(BT, BS),
            order=(1, 0)
        )
        z_block_ptr = tl.make_block_ptr(
            base=z_ptr + i_bh * s_s_h,
            shape=(T, S),
            strides=(s_s_t, s_s_d),
            offsets=(block_start, i_s * BS),
            block_shape=(BT, BS),
            order=(1, 0)
        )
        
        b_s = tl.load(s_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        block_cumsum = tl.dot(m_s, b_s, allow_tf32=False)
        block_result = block_cumsum + b_z[None, :]
        tl.store(z_block_ptr, block_result.to(z_block_ptr.dtype.element_ty), boundary_check=(0, 1))
        
        b_z += tl.sum(b_s, axis=0)

def chunk_global_reversed_cumsum_vector(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T, S = s.shape
    BS = 32
    
    dtype = dtype or s.dtype
    z = torch.empty_like(s, dtype=dtype)
    grid = (triton.cdiv(S, BS), B * H)
    
    chunk_global_reversed_cumsum_vector_kernel[grid](
        s, z,
        s.stride(1), s.stride(2), s.stride(3),
        T=T, S=S, BS=BS
    )
    return z
