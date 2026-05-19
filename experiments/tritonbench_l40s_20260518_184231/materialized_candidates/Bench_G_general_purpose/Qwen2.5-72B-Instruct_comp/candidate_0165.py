import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, row_stride, n_cols, mask_ptr, BLOCK_SIZE: tl.constexpr):
    # Identify the row index
    row_idx = tl.program_id(0)
    # Compute the starting pointer for the row
    row_start_ptr = input_ptr + row_idx * row_stride
    # Define the block of columns to process
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the input row into SRAM
    input_row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Compute the maximum value in the row for numerical stability
    max_val = tl.max(input_row, axis=0)
    # Adjust the input row by subtracting the maximum value
    input_row = input_row - max_val
    # Optionally add a mask
    if mask_ptr is not None:
        mask_row = tl.load(mask_ptr + row_idx * row_stride + col_offsets, mask=col_offsets < n_cols, other=0)
        input_row = input_row + mask_row
    # Compute the exponentials
    exp_row = tl.exp(input_row)
    # Compute the sum of the exponentials (denominator)
    sum_exp = tl.sum(exp_row, axis=0)
    # Compute the softmax values
    output_row = exp_row / sum_exp
    # Store the output row back in the output tensor
    output_ptrs = output_ptr + row_idx * row_stride + col_offsets
    tl.store(output_ptrs, output_row, mask=col_offsets < n_cols)

import torch
import triton
import triton.language as tl

def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
    # Validate the input tensor and mask dimensions
    if dim != -1:
        raise ValueError("Currently, only the last dimension is supported for softmax.")
    if mask is not None and mask.shape != input.shape:
        raise ValueError("Mask tensor must have the same shape as the input tensor.")
    
    # Reshape the input tensor into 2D if necessary
    original_shape = input.shape
    input = input.view(-1, input.shape[-1])
    
    # Get the number of rows and columns
    n_rows, n_cols = input.shape
    
    # Set up the grid and block sizes
    BLOCK_SIZE = 128  # Block size for the kernel
    grid = (n_rows, )
    
    # Allocate the output tensor
    output = torch.empty_like(input)
    
    # Launch the kernel
    if mask is not None:
        mask = mask.view(-1, mask.shape[-1])
        softmax_kernel[grid](output, input, n_cols, n_cols, mask, BLOCK_SIZE)
    else:
        softmax_kernel[grid](output, input, n_cols, n_cols, None, BLOCK_SIZE)
    
    # Reshape the output tensor back to the original shape
    output = output.view(original_shape)
    
    return output
