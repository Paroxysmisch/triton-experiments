import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    stride_row,  # Stride between rows in the input tensor
    stride_col,  # Stride between columns in the input tensor
    n_cols,  # Number of columns in the input tensor
    BLOCK_SIZE: tl.constexpr  # Block size (number of elements processed per program instance)
):
    # Get the row index for this program instance
    row_idx = tl.program_id(0)
    
    # Initialize the output row with zeros
    row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load the row data from the input tensor
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols - i
        row[cols] = tl.load(input_ptr + row_idx * stride_row + (i + cols) * stride_col, mask=mask)
    
    # Compute the maximum value in the row for numerical stability
    max_val = tl.max(row, axis=0)
    
    # Subtract the maximum value from each element in the row
    row -= max_val
    
    # Compute the exponential of each element
    row = tl.exp(row)
    
    # Compute the sum of the exponentials
    sum_exp = tl.sum(row, axis=0)
    
    # Normalize the row by dividing by the sum of exponentials
    row /= sum_exp
    
    # Write the result back to the output tensor
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols - i
        tl.store(output_ptr + row_idx * stride_row + (i + cols) * stride_col, row[cols], mask=mask)

import torch

def softmax(input_tensor: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    input_tensor = input_tensor.cuda()
    
    # Get the dimensions of the input tensor
    n_rows, n_cols = input_tensor.shape
    
    # Allocate memory for the output tensor
    output_tensor = torch.empty_like(input_tensor, device=input_tensor.device)
    
    # Define the grid and block dimensions
    grid = (n_rows, 1, 1)
    block = (1024, 1, 1)  # Adjust block size as needed
    
    # Launch the kernel
    softmax_kernel[grid, block](
        input_tensor,  # Pointer to the input tensor
        output_tensor,  # Pointer to the output tensor
        input_tensor.stride(0),  # Stride between rows in the input tensor
        input_tensor.stride(1),  # Stride between columns in the input tensor
        n_cols,  # Number of columns in the input tensor
        BLOCK_SIZE=block[0]  # Block size
    )
    
    return output_tensor
