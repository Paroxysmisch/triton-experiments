import triton
import triton.language as tl

@triton.jit
def min_kernel(
    input_ptr, min_ptr, min_indices_ptr,
    n_elements, dim_size, dim_stride,
    BLOCK_SIZE: tl.constexpr
):
    # Define the range of elements this program will process
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # Initialize min values and indices
    min_val = tl.full([BLOCK_SIZE], float('inf'), tl.float32)
    min_idx = tl.full([BLOCK_SIZE], -1, tl.int32)

    for i in range(dim_size):
        # Calculate the index for the current element
        idx = offset + i * dim_stride
        # Load the current element
        val = tl.load(input_ptr + idx, mask=mask, other=float('inf'))
        # Update min value and index if the current value is less
        min_val = tl.where(val < min_val, val, min_val)
        min_idx = tl.where(val < min_val, i, min_idx)

    # Store the result
    tl.store(min_ptr + offset, min_val, mask=mask)
    tl.store(min_indices_ptr + offset, min_idx, mask=mask)

import torch

def min(input, dim, keepdim=False, *, out=None):
    # Validate input tensor
    assert isinstance(input, torch.Tensor), "Input must be a torch.Tensor"
    assert isinstance(dim, int), "Dimension must be an integer"

    # Get the shape of the input tensor
    shape = input.shape
    num_elements = shape[dim]
    dim_stride = input.stride(dim)

    # Calculate the number of blocks
    BLOCK_SIZE = 1024
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Prepare output tensors
    if out is None:
        min_values = torch.empty_like(input).select(dim, 0)
        min_indices = torch.empty_like(input, dtype=torch.long).select(dim, 0)
    else:
        min_values, min_indices = out

    # Launch the Triton kernel
    min_kernel[(num_blocks,)](
        input.data_ptr(), min_values.data_ptr(), min_indices.data_ptr(),
        num_elements, shape[dim], dim_stride,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Handle keepdim
    if not keepdim:
        min_values = min_values.squeeze(dim)
        min_indices = min_indices.squeeze(dim)

    return min_values, min_indices
