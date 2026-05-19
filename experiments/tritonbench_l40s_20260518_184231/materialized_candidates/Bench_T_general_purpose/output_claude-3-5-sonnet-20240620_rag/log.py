import torch
import triton
import triton.language as tl

@triton.jit
def log_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load the row into SRAM
    row = tl.load(input_ptrs, mask=col_offsets < n_cols)
    
    # Compute the natural logarithm
    log_output = tl.log(row)
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, log_output, mask=col_offsets < n_cols)

def log(input, *, out=None):
    n_rows, n_cols = input.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Allocate output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Enqueue kernel
    log_kernel[(n_rows,)](
        out,
        input,
        input.stride(0),
        out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
