import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,  # Pointer to the output tensor
    input_ptr,   # Pointer to the input tensor
    mask_ptr,    # Pointer to the mask tensor (optional)
    n_cols,      # Number of columns in the input tensor
    stride,      # Stride between rows in the input tensor
    block_size: tl.constexpr,  # Block size for parallelization
):
    # Compute the row index for the current program instance
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, block_size)
    input_offsets = row_idx * stride + col_offsets

    # Load the input row
    input_row = tl.load(input_ptr + input_offsets, mask=col_offsets < n_cols, other=-float('inf'))

    # Compute the maximum value in the row for numerical stability
    max_val = tl.max(input_row, axis=0)

    # Subtract the maximum value from the input row
    input_row = input_row - max_val

    # Apply the mask if provided
    if mask_ptr is not None:
        mask_row = tl.load(mask_ptr + input_offsets, mask=col_offsets < n_cols, other=0)
        input_row = input_row * mask_row

    # Compute the exponentials
    exp_row = tl.exp(input_row)

    # Compute the sum of the exponentials
    sum_exp = tl.sum(exp_row, axis=0)

    # Normalize the exponentials to get the softmax values
    softmax_row = exp_row / sum_exp

    # Store the result in the output tensor
    tl.store(output_ptr + input_offsets, softmax_row, mask=col_offsets < n_cols)

import torch
import triton
import triton.language as tl

def softmax(input_tensor, mask_tensor=None):
    # Validate input tensor
    if input_tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D (rows x cols)")
    
    # Validate mask tensor if provided
    if mask_tensor is not None:
        if mask_tensor.dim() != 2 or mask_tensor.shape != input_tensor.shape:
            raise ValueError("Mask tensor must be 2D and have the same shape as the input tensor")
    
    # Get the shape of the input tensor
    n_rows, n_cols = input_tensor.shape

    # Allocate the output tensor
    output_tensor = torch.empty_like(input_tensor, device=input_tensor.device)

    # Define the block size
    block_size = 128

    # Configure the grid
    grid = (n_rows,)

    # Launch the kernel
    softmax_kernel[grid](
        output_tensor,  # Pointer to the output tensor
        input_tensor,   # Pointer to the input tensor
        mask_tensor,    # Pointer to the mask tensor (optional)
        n_cols,         # Number of columns in the input tensor
        n_cols,         # Stride between rows in the input tensor
        block_size,     # Block size for parallelization
    )

    return output_tensor
