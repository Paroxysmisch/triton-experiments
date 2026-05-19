import torch
import triton
import triton.language as tl
import math

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row_idx = tl.program_id(axis=0)
    
    # Compute memory offsets for this row
    row_start_in = row_idx * input_row_stride
    row_start_out = row_idx * output_row_stride
    
    # Create a mask for valid columns
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load input row with masking
    row = tl.load(input_ptr + row_start_in + col_offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Subtract max and compute exponentials
    numerator = tl.exp(row - row_max)
    
    # Compute sum for normalization
    denominator = tl.sum(numerator, axis=0)
    
    # Normalize
    softmax_output = numerator / denominator
    
    # Store the result
    tl.store(output_ptr + row_start_out + col_offsets, softmax_output, mask=mask)

def triton_softmax(x):
    # Get input dimensions
    n_rows, n_cols = x.shape
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Compute block size (next power of 2, capped at 1024)
    BLOCK_SIZE = min(1024, 2 ** math.ceil(math.log2(n_cols)))
    
    # Configure grid
    grid = (n_rows,)
    
    # Launch kernel
    softmax_kernel[grid](
        x,
        output,
        x.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
