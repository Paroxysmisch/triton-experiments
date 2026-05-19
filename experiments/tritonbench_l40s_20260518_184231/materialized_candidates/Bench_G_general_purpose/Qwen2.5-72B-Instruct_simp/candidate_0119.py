import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,  # Pointer to the output matrix
    input_ptr,   # Pointer to the input matrix
    input_row_stride,  # Row stride of the input matrix
    output_row_stride,  # Row stride of the output matrix
    n_cols,      # Number of columns in the matrix
    BLOCK_SIZE: tl.constexpr  # Block size
):
    # Get the row index
    row_idx = tl.program_id(axis=0)
    
    # Compute the starting and ending column indices for this block
    col_start = tl.arange(0, BLOCK_SIZE)
    col_end = tl.minimum(n_cols, col_start + BLOCK_SIZE)
    
    # Load the row into SRAM
    row = tl.load(input_ptr + row_idx * input_row_stride + col_start, mask=col_start < n_cols)
    
    # Find the maximum value in the row for numerical stability
    max_val = tl.max(row, axis=0)
    
    # Compute the numerator (exp(x - max_val))
    numerator = tl.exp(row - max_val)
    
    # Compute the denominator (sum of the numerator)
    denominator = tl.sum(numerator, axis=0)
    
    # Compute the softmax values
    softmax_values = numerator / denominator
    
    # Store the result back to the output matrix
    tl.store(output_ptr + row_idx * output_row_stride + col_start, softmax_values, mask=col_start < n_cols)

import triton
import triton.language as tl

def triton_softmax(input_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE=128):
    # Determine the grid size
    grid = (n_rows, )
    
    # Launch the kernel
    softmax_kernel[grid](
        output_ptr,  # Pointer to the output matrix
        input_ptr,   # Pointer to the input matrix
        n_cols,      # Row stride of the input matrix (same as number of columns)
        n_cols,      # Row stride of the output matrix (same as number of columns)
        n_cols,      # Number of columns in the matrix
        BLOCK_SIZE   # Block size
    )
