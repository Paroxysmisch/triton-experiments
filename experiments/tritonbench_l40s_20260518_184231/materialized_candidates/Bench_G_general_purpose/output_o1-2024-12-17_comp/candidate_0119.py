import triton
import triton.language as tl
import torch
import math

@triton.jit
def softmax_kernel(
    input_ptr, 
    output_ptr, 
    input_row_stride, 
    output_row_stride, 
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    input_offset = row_id * input_row_stride + cols
    output_offset = row_id * output_row_stride + cols

    mask = cols < n_cols

    # Load row
    x = tl.load(input_ptr + input_offset, mask=mask, other=-float('inf'))

    # Compute max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max

    # Exponentiate
    x_exp = tl.exp(x)

    # Sum of exponentiated values
    x_sum = tl.sum(x_exp, axis=0)

    # Normalize
    x_softmax = x_exp / x_sum

    # Store result
    tl.store(output_ptr + output_offset, x_softmax, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2, "Input tensor must be 2D"
    n_rows, n_cols = x.shape
    # Compute BLOCK_SIZE as next power of 2, capped at 1024
    block_size = 2 ** ((n_cols - 1).bit_length())
    block_size = min(block_size, 1024)

    y = torch.empty_like(x)
    grid = (n_rows,)

    softmax_kernel[grid](
        x, 
        y, 
        x.stride(0), 
        y.stride(0), 
        n_cols,
        BLOCK_SIZE=block_size
    )
    return y
