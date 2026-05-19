import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Identify the current row using program_id(0)
    row_id = tl.program_id(0)
    
    # Calculate the starting pointer for this row
    input_row_start = input_ptr + row_id * input_row_stride
    output_row_start = output_ptr + row_id * output_row_stride
    
    # Load the row into on-chip memory with masking
    row = tl.load(input_row_start + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=-float('inf'))
    
    # Numerical stabilization: subtract the maximum value from the row elements
    max_val = tl.max(row, axis=0)
    row = row - max_val
    
    # Compute the exponentials
    exp_row = tl.exp(row)
    
    # Compute the sum of exponentials
    sum_exp = tl.sum(exp_row, axis=0)
    
    # Normalize to derive the softmax probabilities
    softmax_row = exp_row / sum_exp
    
    # Store the result back in the output matrix
    tl.store(output_row_start + tl.arange(0, BLOCK_SIZE), softmax_row, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

import torch

def softmax(input_tensor: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    input_tensor = input_tensor.cuda()
    
    # Get the number of rows and columns
    n_rows, n_cols = input_tensor.shape
    
    # Calculate the optimal BLOCK_SIZE (smallest power of two greater than n_cols)
    BLOCK_SIZE = 1
    while BLOCK_SIZE < n_cols:
        BLOCK_SIZE *= 2
    
    # Adjust num_warps based on BLOCK_SIZE
    num_warps = 4 if BLOCK_SIZE <= 256 else 8
    
    # Initialize an output tensor
    output_tensor = torch.empty_like(input_tensor)
    
    # Launch the Triton kernel with one block per input matrix row
    grid = (n_rows, )
    softmax_kernel[grid](input_tensor, output_tensor, input_tensor.stride(0), output_tensor.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    
    return output_tensor
