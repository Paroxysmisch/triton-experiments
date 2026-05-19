import triton
import triton.language as tl

@triton.jit
def argmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    
    max_val = tl.max(row)
    max_indices = tl.where(row == max_val, col_offsets, -1)
    
    # Reduce across the block to find the global max index
    max_val = tl.max_contiguous(max_val, axis=0)
    max_indices = tl.where(max_val == max_indices[0], max_indices[0], -1)
    
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, max_indices, mask=col_offsets < n_cols)
