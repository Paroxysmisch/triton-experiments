@triton.jit
def min_kernel(min_out_ptr, min_index_out_ptr, input_ptr, input_row_stride, input_col_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offset = tl.arange(0, BLOCK_SIZE)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Load the row into shared memory
    row = tl.load(row_start_ptr + col_offset * input_col_stride, mask=col_offset < n_cols, other=float('inf'))

    # Find the minimum value and its index
    min_val = tl.min(row)
    min_idx = tl.argmin(row)

    # Write the results to global memory
    min_out_ptr[row_idx] = min_val
    min_index_out_ptr[row_idx] = min_idx
