import triton
import triton.language as tl
import torch
import math

@triton.jit
def square_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_row_stride,  # Stride between rows in input
    output_row_stride,  # Stride between rows in output
    n_cols,  # Number of columns
    BLOCK_SIZE: tl.constexpr,  # Power of 2 >= n_cols for memory layout
):
    # Get the row index for this kernel instance
    row_idx = tl.program_id(0)
    
    # Calculate starting pointers for this row
    row_start_in = input_ptr + row_idx * input_row_stride
    row_start_out = output_ptr + row_idx * output_row_stride
    
    # Create offsets for the columns
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid columns
    mask = col_offsets < n_cols
    
    # Load the row into SRAM
    row = tl.load(row_start_in + col_offsets, mask=mask)
    
    # Compute square
    square_output = row * row
    
    # Store the result back to global memory
    tl.store(row_start_out + col_offsets, square_output, mask=mask)

def square(x: torch.Tensor) -> torch.Tensor:
    # Get input dimensions
    n_rows, n_cols = x.shape
    
    # Calculate BLOCK_SIZE as next power of 2 after n_cols
    BLOCK_SIZE = 2 ** math.ceil(math.log2(n_cols))
    
    # Determine number of warps based on BLOCK_SIZE
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 1024:
        num_warps = 6
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Launch kernel
    square_kernel[(n_rows,)](
        x.data_ptr(),
        y.data_ptr(),
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return y
