import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    input_row_stride,  # Stride for input rows
    output_row_stride,  # Stride for output rows
    n_cols,  # Number of columns in the row being processed
    BLOCK_SIZE: tl.constexpr  # Block size for parallel processing
):
    # Compute the row index
    row_id = tl.program_id(axis=0)
    
    # Compute the memory offsets for the current row
    input_row_offset = row_id * input_row_stride
    output_row_offset = row_id * output_row_stride
    
    # Load the row of data from global memory into on-chip SRAM
    row = tl.load(input_ptr + input_row_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=-float('inf'))
    
    # Compute the maximum value in the row for numerical stability
    max_val = tl.max(row, axis=0)
    
    # Subtract the maximum value from each element in the row
    row_sub = row - max_val
    
    # Compute the exponentiation to form the numerator
    numerator = tl.exp(row_sub)
    
    # Compute the sum of the numerators to serve as the denominator
    denominator = tl.sum(numerator, axis=0)
    
    # Compute the softmax values
    softmax_values = numerator / denominator
    
    # Store the result back to the output tensor
    tl.store(output_ptr + output_row_offset + tl.arange(0, BLOCK_SIZE), softmax_values, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

import torch
import triton
import triton.language as tl

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    # Extract dimensions of the input tensor
    n_rows, n_cols = x.shape
    
    # Initialize the output tensor of the same shape
    y = torch.empty_like(x)
    
    # Determine the block size for parallel processing
    block_size = min(1024, max(1, 2 ** (n_cols - 1).bit_length()))
    
    # Configure the execution grid to cover all rows
    grid = (n_rows,)
    
    # Launch the kernel
    softmax_kernel[grid](
        x,  # Input tensor
        y,  # Output tensor
        x.stride(0),  # Input row stride
        y.stride(0),  # Output row stride
        n_cols,  # Number of columns
        BLOCK_SIZE=block_size  # Block size for parallel processing
    )
    
    return y
