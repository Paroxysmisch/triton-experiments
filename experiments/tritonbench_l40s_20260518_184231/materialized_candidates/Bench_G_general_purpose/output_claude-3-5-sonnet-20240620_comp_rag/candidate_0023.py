import triton
import triton.language as tl
import torch
import math

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # Get the row index we're processing
    row_idx = tl.program_id(0)
    
    # Calculate start pointers for this row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Create offsets for this block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load row values using mask for when BLOCK_SIZE > n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    
    # Convert to float32 for better precision
    row_f32 = row.to(tl.float32)
    
    # Numerical stability: subtract maximum
    row_max = tl.max(row_f32, axis=0)
    row_minus_max = row_f32 - row_max
    
    # Compute exponentials and sum
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    
    # Compute softmax output
    softmax_output = numerator / denominator
    
    # Store results
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output.to(row.dtype), mask=col_offsets < n_cols)

def softmax(x):
    """
    Compute softmax over the last dimension of x
    """
    # Extract input shape and verify last dimension
    *batch_dims, n_cols = x.shape
    n_rows = 1
    for dim in batch_dims:
        n_rows *= dim
    
    # Reshape input to 2D tensor
    x_2d = x.reshape(n_rows, n_cols)
    
    # Compute output tensor
    output = torch.empty_like(x_2d)
    
    # Calculate optimal block size (next power of 2 >= n_cols)
    BLOCK_SIZE = 2 ** math.ceil(math.log2(n_cols))
    
    # Calculate optimal number of warps
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    # Launch kernel
    softmax_kernel[(n_rows,)](
        output, x_2d,
        x_2d.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    # Reshape output back to original dimensions
    return output.reshape(*batch_dims, n_cols)
