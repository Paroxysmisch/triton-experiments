import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,  # Pointer to output tensor
    input_ptr,   # Pointer to input tensor
    input_row_stride,  # Input row stride
    output_row_stride, # Output row stride
    n_rows,      # Number of rows 
    n_cols,      # Number of columns
    BLOCK_SIZE: tl.constexpr,  # Static block size for processing
    num_stages: tl.constexpr,  # Number of pipeline stages
):
    # Get program ID and total number of programs
    row_idx = tl.program_id(0)
    
    # Compute pointers to input/output row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # Create offsets for the block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load input row with mask for handling boundary conditions
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Compute exponentials and sum
    exp_row = tl.exp(row - row_max)
    exp_sum = tl.sum(exp_row, axis=0)
    
    # Normalize and store result
    output = exp_row / exp_sum
    tl.store(out_row_start_ptr + col_offsets, output, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Apply softmax to input tensor along last dimension
    Args:
        x: Input tensor of shape (batch_size, n_features)
    Returns:
        Output tensor with same shape as input
    """
    # Handle input validation and shape
    assert x.is_cuda and x.is_contiguous()
    n_rows, n_cols = x.shape
    
    # Compute optimal block size (next power of 2)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK_SIZE <= 256 else 8
    num_stages = 2
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Launch kernel
    grid = (n_rows,)
    softmax_kernel[grid](
        output_ptr=output,
        input_ptr=x,
        input_row_stride=x.stride(0),
        output_row_stride=output.stride(0),
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    
    return output
