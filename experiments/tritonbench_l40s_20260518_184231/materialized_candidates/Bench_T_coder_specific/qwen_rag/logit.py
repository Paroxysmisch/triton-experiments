import triton
import triton.language as tl

@triton.jit
def logit_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    # The rows are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=float('nan'))
    
    # Clamp the input
    z = tl.where(row < eps, eps, row)
    z = tl.where(z > 1 - eps, 1 - eps, z)
    
    # Compute the logit
    logit_output = tl.log(z / (1 - z))
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, logit_output, mask=col_offsets < n_cols)
