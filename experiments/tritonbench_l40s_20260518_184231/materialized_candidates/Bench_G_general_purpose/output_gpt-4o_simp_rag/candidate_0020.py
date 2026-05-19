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
    s_ptr,  # input tensor
    z_ptr,  # output tensor
    stride_h, stride_t,  # strides for input tensor
    T: tl.constexpr,  # total size of the last dimension
    BT: tl.constexpr  # block size for the last dimension
):
    # Get program ids for batch and head
    i_bh = tl.program_id(0)
    
    # Indices for the chunk within the last dimension
    o_i = tl.arange(0, BT)
    
    # Initialize the cumulative sum for the chunk
    b_z = tl.zeros([BT], dtype=tl.float32)
    
    # Iterate backwards over the last dimension in chunks
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        # Calculate the pointer to the current block in the input and output
        p_s = s_ptr + i_bh * stride_h + (i_t * BT) * stride_t
        p_z = z_ptr + i_bh * stride_h + (i_t * BT) * stride_t
        
        # Load the block of input data
        b_s = tl.load(p_s + o_i * stride_t, mask=o_i < T - i_t * BT, other=0.0).to(tl.float32)
        
        # Update the cumulative sum
        b_z += b_s
        
        # Store the result
        tl.store(p_z + o_i * stride_t, b_z, mask=o_i < T - i_t * BT)

def chunk_global_reversed_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T = s.shape
    BT = 32  # You can choose an appropriate block size

    dtype = dtype or s.dtype
    grid = (B * H,)
    z = torch.empty_like(s, dtype=dtype)
    chunk_global_reversed_cumsum_scalar_kernel[grid](
        s, z,
        s.stride(1), s.stride(2),
        T=T, BT=BT
    )
    return z

# Example usage
s = torch.randn(4, 8, 128, device='cuda', dtype=torch.float32)
z = chunk_global_reversed_cumsum_scalar(s)
