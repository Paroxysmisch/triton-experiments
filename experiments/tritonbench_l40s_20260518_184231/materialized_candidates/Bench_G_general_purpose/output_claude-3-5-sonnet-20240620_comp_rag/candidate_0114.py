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
    ],
    key=['T']
)
@triton.jit
def chunk_global_cumsum_scalar_kernel(
    s,  # input tensor
    o,  # output tensor
    stride_h,  # stride for head dimension
    stride_t,  # stride for time dimension
    T: tl.constexpr,  # sequence length
    BT: tl.constexpr,  # block size for time dimension
):
    # Get program ID for batch-head combination
    pid = tl.program_id(0)
    
    # Initialize running sum
    running_sum = tl.zeros([1], dtype=tl.float32)
    
    # Create offset indices for the block
    offsets = tl.arange(0, BT)
    
    # Process sequence in chunks
    for t_start in range(0, T, BT):
        # Create block pointers for current chunk
        block_ptr_in = tl.make_block_ptr(
            base=s + pid * stride_h,
            shape=(T,),
            strides=(stride_t,),
            offsets=(t_start,),
            block_shape=(BT,),
            order=(0,)
        )
        
        block_ptr_out = tl.make_block_ptr(
            base=o + pid * stride_h,
            shape=(T,),
            strides=(stride_t,),
            offsets=(t_start,),
            block_shape=(BT,),
            order=(0,)
        )
        
        # Load input block
        x = tl.load(block_ptr_in, boundary_check=(0,))
        x = x.to(tl.float32)
        
        # Compute cumsum within block
        block_sum = tl.cumsum(x)
        
        # Add running sum to all elements
        block_sum = block_sum + running_sum
        
        # Store result
        tl.store(block_ptr_out, block_sum, boundary_check=(0,))
        
        # Update running sum
        running_sum += tl.sum(x)

def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    # Handle optional dtype
    dtype = dtype or s.dtype
    
    # Get input dimensions
    B, H, T = s.shape
    
    # Create output tensor
    z = torch.empty_like(s, dtype=dtype)
    
    # Calculate grid size (one instance per batch-head combination)
    grid = (B * H,)
    
    # Launch kernel
    chunk_global_cumsum_scalar_kernel[grid](
        s, z,
        s.stride(1),  # head stride
        s.stride(2),  # time stride
        T=T,
        BT=32,  # can be autotuned
    )
    
    return z
