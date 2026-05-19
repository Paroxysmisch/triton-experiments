import torch
import triton
import triton.language as tl
import math

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    stride_row, stride_col,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    row_idx = tl.program_id(0)
    
    # Compute the row offset for input and output
    row_start_ptr = input_ptr + row_idx * stride_row
    
    # Create offsets for the current block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load input elements
    row = tl.load(row_start_ptr + col_offsets * stride_col, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)
    
    # Compute exponentials
    numerator = tl.exp(row - row_max)
    
    # Compute sum for denominator
    denominator = tl.sum(numerator, axis=0)
    
    # Compute softmax
    softmax_output = numerator / denominator
    
    # Store the result
    output_row_start_ptr = output_ptr + row_idx * stride_row
    tl.store(output_row_start_ptr + col_offsets * stride_col, softmax_output, mask=mask)

def softmax(x):
    # Get input dimensions
    batch_size, n_cols = x.shape
    
    # Compute block size (next power of 2)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Allocate output
    y = torch.empty_like(x)
    
    # Configure grid
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    # Enqueue kernel
    softmax_kernel[(batch_size,)](
        y, x,
        x.stride(0), x.stride(1),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return y
