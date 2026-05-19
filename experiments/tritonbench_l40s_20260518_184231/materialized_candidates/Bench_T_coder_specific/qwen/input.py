import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    x_ptr,          # Pointer to input tensor
    y_ptr,          # Pointer to output tensor
    n_elements,     # Total number of elements in the input tensor
    n_rows,         # Number of rows in the input tensor
    n_cols,         # Number of columns in the input tensor
    block_size: tl.constexpr,  # Block size for parallel execution
):
    row = tl.program_id(0)
    col = tl.program_id(1)

    if row >= n_rows:
        return

    # Initialize sum and count for the current row
    row_sum = 0.0
    count = 0

    # Iterate over columns to compute the sum and count for the current row
    for c in range(col, n_cols, block_size):
        col_val = tl.load(x_ptr + row * n_cols + c)
        row_sum += col_val
        count += 1

    # Perform reduction across blocks within the same row
    row_sum = tl.block_reduce(row_sum, block_size)
    count = tl.block_reduce(count, block_size)

    # Write the result to the output tensor
    if tl.thread_idx[0] == 0:
        y_ptr[row] = row_sum / count
