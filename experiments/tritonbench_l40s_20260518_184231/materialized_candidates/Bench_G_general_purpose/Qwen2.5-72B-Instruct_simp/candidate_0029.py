import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,  # Pointer to the output matrix
    input_ptr,   # Pointer to the input matrix
    input_row_stride,  # Stride between elements in the same row of the input matrix
    output_row_stride,  # Stride between elements in the same row of the output matrix
    n_cols,      # Number of columns in the matrix
    BLOCK_SIZE: tl.constexpr  # Block size for parallel execution
):
    # Get the row index for this thread block
    row_idx = tl.program_id(0)
    
    # Initialize a block of memory to store the row data
    row_data = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load the row data into the block
    for i in range(n_cols):
        row_data[i] = tl.load(input_ptr + row_idx * input_row_stride + i)
    
    # Compute the maximum value in the row for numerical stability
    max_val = tl.max(row_data, axis=0)
    
    # Subtract the maximum value from each element in the row
    row_data = tl.exp(row_data - max_val)
    
    # Compute the sum of the exponentials
    sum_exp = tl.sum(row_data, axis=0)
    
    # Normalize the row data
    row_data = row_data / sum_exp
    
    # Store the normalized row data back into the output matrix
    for i in range(n_cols):
        tl.store(output_ptr + row_idx * output_row_stride + i, row_data[i])

import torch
import triton

def softmax(input_tensor: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    assert input_tensor.is_cuda, "Input tensor must be on the GPU"
    
    # Get the dimensions of the input tensor
    n_rows, n_cols = input_tensor.shape
    
    # Determine the block size and number of warps
    BLOCK_SIZE = 128  # Adjust this based on your GPU architecture
    num_warps = 4  # Adjust this based on your GPU architecture
    
    # Allocate space for the output tensor
    output_tensor = torch.empty_like(input_tensor)
    
    # Launch the kernel
    softmax_kernel[torch.arange(n_rows), (BLOCK_SIZE,)](
        output_tensor,  # Pointer to the output matrix
        input_tensor,   # Pointer to the input matrix
        n_cols,         # Number of columns in the matrix
        n_cols,         # Stride between elements in the same row (same as n_cols for a contiguous matrix)
        n_cols,         # Number of columns in the matrix
        BLOCK_SIZE      # Block size for parallel execution
    )
    
    return output_tensor
