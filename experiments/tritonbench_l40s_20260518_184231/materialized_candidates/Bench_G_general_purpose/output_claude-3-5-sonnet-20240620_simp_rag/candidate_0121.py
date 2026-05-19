import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def chunk_global_cumsum_scalar_kernel(
    # Pointers to tensors
    s_ptr,          # Input tensor pointer
    o_ptr,          # Output tensor pointer
    # Tensor strides
    stride_h,       # Head dimension stride
    stride_t,       # Time dimension stride 
    stride_d,       # Feature dimension stride
    # Shape parameters
    T: tl.constexpr,  # Time dimension size
    D: tl.constexpr,  # Feature dimension size
    BT: tl.constexpr, # Block size for time dimension
    BD: tl.constexpr  # Block size for feature dimension
):
    # Get program ID for feature and batch*head dimensions
    pid_f = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # Create mask for cumsum operation
    offs_t = tl.arange(0, BT)
    mask = tl.where(offs_t[:, None] >= offs_t[None, :], 1.0, 0.0)
    
    # Initialize running sum for the block
    running_sum = tl.zeros([BD], dtype=tl.float32)
    
    # Iterate over time dimension in blocks
    for t_start in range(0, T, BT):
        # Create block pointers
        block_s = tl.make_block_ptr(
            s_ptr + pid_bh * stride_h,
            (T, D),
            (stride_t, stride_d),
            (t_start, pid_f * BD),
            (BT, BD),
            (1, 0)
        )
        block_o = tl.make_block_ptr(
            o_ptr + pid_bh * stride_h,
            (T, D),
            (stride_t, stride_d),
            (t_start, pid_f * BD),
            (BT, BD),
            (1, 0)
        )
        
        # Load input block
        block_data = tl.load(block_s, boundary_check=(0, 1)).to(tl.float32)
        
        # Compute cumsum within block
        block_cumsum = running_sum[None, :] + tl.dot(mask, block_data, allow_tf32=False)
        
        # Store result
        tl.store(block_o, block_cumsum.to(block_o.dtype.element_ty), boundary_check=(0, 1))
        
        # Update running sum
        running_sum += tl.sum(block_data, 0)

def chunk_global_cumsum_scalar(
    x: torch.Tensor,
    block_size: int = 32
) -> torch.Tensor:
    # Get tensor dimensions
    *batch_dims, T, D = x.shape
    batch_size = 1
    for d in batch_dims:
        batch_size *= d
        
    # Create output tensor
    output = torch.empty_like(x)
    
    # Launch kernel
    grid = (triton.cdiv(D, block_size), batch_size)
    chunk_global_cumsum_scalar_kernel[grid](
        x.data_ptr(),
        output.data_ptr(),
        x.stride(-3) if len(batch_dims) > 1 else 0,  # Head stride
        x.stride(-2),                                # Time stride
        x.stride(-1),                                # Feature stride
        T=T,
        D=D,
        BT=block_size,
        BD=block_size
    )
    
    return output
