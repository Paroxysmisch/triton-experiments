import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr: tl.tensor, output_ptr: tl.tensor,
    input_row_stride: tl.int32, output_row_stride: tl.int32,
    n_cols: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row = row[:n_cols]
    
    # Compute row offsets
    input_row_offset = row * input_row_stride
    output_row_offset = row * output_row_stride
    
    # Load row into shared memory
    input_row = tl.load(input_ptr + input_row_offset, mask=row < n_cols, other=-float('inf'))
    
    # Compute maximum for numerical stability
    max_val = tl.max(input_row)
    input_row = input_row - max_val
    
    # Compute exponentials
    exp_row = tl.exp(input_row)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_row)
    
    # Normalize
    softmax_row = exp_row / sum_exp
    
    # Store result back to output tensor
    tl.store(output_ptr + output_row_offset, softmax_row, mask=row < n_cols)

@triton.jit
def triton_softmax(x: tl.tensor, y: tl.tensor):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = min(1024, 2 ** (n_cols - 1).bit_length())
    
    # Configure the execution grid
    grid = (triton.cdiv(n_rows, BLOCK_SIZE), 1)
    
    # Launch the kernel
    softmax_kernel[grid](x, y, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE)
