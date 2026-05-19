import torch
import triton
import triton.language as tl

@triton.jit
def square_kernel(
    input_ptr,    # Pointer to input tensor
    output_ptr,   # Pointer to output tensor
    input_row_stride,  # Stride between rows of input
    output_row_stride, # Stride between rows of output
    n_cols,       # Number of columns in input
    BLOCK_SIZE: tl.constexpr,  # Static block size for memory operations
):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Calculate starting pointers for this row
    row_start_in = input_ptr + row_idx * input_row_stride
    row_start_out = output_ptr + row_idx * output_row_stride
    
    # Create offsets for the columns
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid columns
    mask = col_offsets < n_cols
    
    # Load the row into SRAM
    row = tl.load(row_start_in + col_offsets, mask=mask)
    
    # Compute square
    square_output = row * row
    
    # Store the result back to global memory
    tl.store(row_start_out + col_offsets, square_output, mask=mask)

def square(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 2, "Input tensor must be 2D"
    
    n_rows, n_cols = x.shape
    
    # Calculate BLOCK_SIZE as the next power of 2 after n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Adjust num_warps based on BLOCK_SIZE for better performance
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 4096:
        num_warps = 16
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Launch kernel
    square_kernel[(n_rows,)](
        x,
        y,
        x.stride(0),  # Input row stride
        y.stride(0),  # Output row stride
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return y
