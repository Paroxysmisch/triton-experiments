import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, row_stride, n_cols, mask_ptr, BLOCK_SIZE: tl.constexpr):
    # Define the row and column indices
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)

    # Compute the pointer to the current row
    input_row_ptr = input_ptr + row_idx * row_stride
    output_row_ptr = output_ptr + row_idx * row_stride

    # Load the input data for the current row
    input_data = tl.load(input_row_ptr + col_idx, mask=col_idx < n_cols, other=-float('inf'))

    # Compute the maximum value for numerical stability
    row_max = tl.max(input_data, axis=0)
    input_data = input_data - row_max

    # Compute the exponentials
    exp_data = tl.exp(input_data)

    # Apply the mask if provided
    if mask_ptr is not None:
        mask_data = tl.load(mask_ptr + row_idx * row_stride + col_idx, mask=col_idx < n_cols, other=0.0)
        exp_data = exp_data * mask_data

    # Compute the sum of exponentials
    sum_exp = tl.sum(exp_data, axis=0)

    # Normalize to get the softmax
    softmax_result = exp_data / sum_exp

    # Store the result
    tl.store(output_row_ptr + col_idx, softmax_result, mask=col_idx < n_cols)

import torch

def softmax(input_tensor, mask_tensor=None):
    # Validate input tensor
    assert input_tensor.ndim == 2, "Input tensor must be 2D"
    n_rows, n_cols = input_tensor.shape

    # Create an output tensor
    output_tensor = torch.empty_like(input_tensor)

    # Define block size (this can be tuned based on your hardware)
    BLOCK_SIZE = 128

    # Configure the grid
    grid = (n_rows,)

    # Launch the Triton kernel
    softmax_kernel[grid](
        output_tensor,
        input_tensor,
        input_tensor.stride(0),
        n_cols,
        mask_tensor if mask_tensor is not None else 0,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output_tensor
