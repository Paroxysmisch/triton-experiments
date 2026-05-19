import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, input_col_stride, output_row_stride, output_col_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Each row starts at a different offset in the input and output tensors
    row_offset_input = row * input_row_stride
    row_offset_output = row * output_row_stride
    
    # Load the entire row into shared memory
    shmem_row = tl.zeros((BLOCK_SIZE,), dtype=input_ptr.dtype)
    shmem_row[col] = tl.load(input_ptr + row_offset_input + col * input_col_stride)
    tl.syncthreads()

    # Compute the maximum for numerical stability
    row_max = tl.max(shmem_row, axis=0)
    
    # Normalize and exponentiate
    exp_values = tl.exp(shmem_row - row_max)
    row_sum = tl.sum(exp_values, axis=0)
    
    # Store the result
    tl.store(output_ptr + row_offset_output + col * output_col_stride, exp_values / row_sum)

def softmax(input, dim, dtype=None):
    assert dim == 1, "Only dim=1 is supported for simplicity"
    n_rows, n_cols = input.shape
    
    # Determine the block size
    BLOCK_SIZE = 128
    
    # Allocate output tensor
    output = torch.empty_like(input)
    
    # Configure the grid and block dimensions
    grid = (n_rows, 1)
    block = (BLOCK_SIZE, 1)
    
    # Launch the kernel
    softmax_kernel[grid, block](output.data_ptr(), input.data_ptr(),
                                 input.stride(0), input.stride(1),
                                 output.stride(0), output.stride(1),
                                 n_rows, n_cols, BLOCK_SIZE)
    
    return output
