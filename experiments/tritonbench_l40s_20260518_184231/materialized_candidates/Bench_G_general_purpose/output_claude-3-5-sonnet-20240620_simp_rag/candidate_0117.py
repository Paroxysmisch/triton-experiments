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
    s,
    z,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    # Get program IDs for spatial and batch*head dimensions
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    
    # Create offset indices and mask matrix
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)
    
    # Initialize accumulator
    b_z = tl.zeros([BS], dtype=tl.float32)
    
    # Iterate through time dimension in reverse
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        # Create block pointers for input and output
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), 
                               (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), 
                               (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        
        # Load input block and compute cumsum
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        
        # Store results
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), boundary_check=(0, 1))
        
        # Update accumulator
        if i_t >= 0:
            b_z += tl.sum(b_s, 0)

def chunk_global_reversed_cumsum_vector(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    B, H, T, S = s.shape
    BS = 32  # Spatial block size
    
    # Prepare output tensor and launch configuration
    dtype = dtype or s.dtype
    grid = (triton.cdiv(S, BS), B * H)
    z = torch.empty_like(s, dtype=dtype)
    
    # Launch kernel
    chunk_global_reversed_cumsum_vector_kernel[grid](
        s, z,
        s.stride(1), s.stride(2), s.stride(3),
        T=T, S=S, BS=BS
    )
    return z
