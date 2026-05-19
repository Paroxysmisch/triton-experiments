import torch
import triton
import triton.language as tl
from typing import Optional

@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 16}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 64}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=4),
    ]
)
@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    s,  # input tensor
    o,  # output tensor
    stride_b,  # stride for batch dimension
    stride_h,  # stride for head dimension 
    stride_t,  # stride for time dimension
    T: tl.constexpr,  # sequence length
    BT: tl.constexpr,  # block size for time dimension
):
    # Program ID for the current instance
    pid_b = tl.program_id(0)  # batch index
    pid_h = tl.program_id(1)  # head index
    
    # Compute base offset for this (batch, head) pair
    base_offset = pid_b * stride_b + pid_h * stride_h
    
    # Initialize accumulator
    acc = tl.zeros([BT], dtype=tl.float32)
    
    # Create mask for valid indices
    offs = tl.arange(0, BT)
    
    # Process chunks in reverse order
    for t in range((T + BT - 1) // BT - 1, -1, -1):
        # Current chunk start position
        chunk_start = t * BT
        
        # Load input block
        block_mask = chunk_start + offs < T
        x = tl.load(s + base_offset + (chunk_start + offs) * stride_t, 
                   mask=block_mask, other=0.0)
        
        # Compute cumulative sum within block
        block_cumsum = acc[None] + tl.cumsum(x, axis=0)
        
        # Store results
        tl.store(o + base_offset + (chunk_start + offs) * stride_t, 
                block_cumsum, mask=block_mask)
        
        # Update accumulator for next iteration
        acc += tl.sum(x, axis=0)

def chunk_global_reversed_cumsum_scalar(
    x: torch.Tensor,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Compute reversed cumulative sum along the last dimension.
    
    Args:
        x: Input tensor of shape (B, H, T)
        dtype: Optional output dtype
    
    Returns:
        Output tensor of shape (B, H, T)
    """
    B, H, T = x.shape
    dtype = dtype or x.dtype
    
    # Create output tensor
    output = torch.empty_like(x, dtype=dtype)
    
    # Launch kernel
    grid = (B, H)
    chunk_global_reversed_cumsum_scalar_kernel[grid](
        x, output,
        x.stride(0), x.stride(1), x.stride(2),
        T=T,
        BT=min(64, T)  # Choose block size based on sequence length
    )
    
    return output
