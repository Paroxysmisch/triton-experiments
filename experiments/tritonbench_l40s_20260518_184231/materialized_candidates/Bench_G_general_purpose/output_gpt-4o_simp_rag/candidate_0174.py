import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Parallelize across the rows of the matrix
    row_idx = tl.program_id(0)
    # Compute the starting pointer for the row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # Generate column offsets for loading the row into SRAM
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, with masking to handle cases where BLOCK_SIZE > n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract the maximum value for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Compute exponentials of the adjusted row
    numerator = tl.exp(row_minus_max)
    # Sum the exponentials
    denominator = tl.sum(numerator, axis=0)
    # Compute the softmax output
    softmax_output = numerator / denominator
    # Write the result back to global memory
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def softmax(x):
    n_rows, n_cols = x.shape
    # Determine the block size as the next power of two greater than the number of columns
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Set the number of warps based on the block size for efficient execution
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output tensor
    y = torch.empty_like(x)
    # Launch the kernel with a 1D grid where each instance processes one row of the matrix
    softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
