import torch
import triton
import triton.language as tl

# Triton kernel for softmax
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

# Wrapper function
def softmax(input, dim, dtype=None) -> torch.Tensor:
    if dtype is not None:
        input = input.to(dtype)
    
    shape = input.shape
    n_rows = shape[dim]
    n_cols = shape[dim + 1] if dim + 1 < len(shape) else 1
    input = input.view(n_rows, -1)
    
    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `input`
    BLOCK_SIZE = triton.next_power_of_2(input.shape[1])
    
    # Allocate output
    output = torch.empty_like(input)
    
    # Number of warps and stages
    num_warps = 4
    num_stages = 2
    
    # Grid configuration
    grid = (n_rows, 1)
    
    # Launch the kernel
    softmax_kernel[grid](
        output_ptr=output.data_ptr(),
        input_ptr=input.data_ptr(),
        input_row_stride=input.stride(0),
        output_row_stride=output.stride(0),
        n_rows=n_rows,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages
    )
    
    # Reshape back to original shape
    output = output.view(shape)
    
    return output
