import triton
import triton.language as tl

@triton.jit
def max_reduce_kernel(input_ptr, output_values_ptr, output_indices_ptr, stride, dim_size, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offset = row_idx * stride

    # Initialize max values and indices
    max_val = tl.full([BLOCK_SIZE], float('-inf'), tl.float32)
    max_idx = tl.zeros([BLOCK_SIZE], tl.int32)

    for i in range(0, n_cols, BLOCK_SIZE):
        # Load a block of data
        idx = tl.arange(0, BLOCK_SIZE) + i
        mask = idx < n_cols
        input_vals = tl.load(input_ptr + offset + idx, mask=mask, other=float('-inf'))

        # Compare and select max
        max_val = tl.where(input_vals > max_val, input_vals, max_val)
        max_idx = tl.where(input_vals > max_val, idx, max_idx)

    # Store the result
    tl.store(output_values_ptr + row_idx, max_val)
    tl.store(output_indices_ptr + row_idx, max_idx)

import torch
from collections import namedtuple

MaxResult = namedtuple('MaxResult', ['values', 'indices'])

def max(input, dim, keepdim=False, *, out=None):
    # Ensure input is a torch.Tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    # Prepare output tensors if not provided
    if out is None:
        if keepdim:
            output_shape = list(input.shape)
            output_shape[dim] = 1
        else:
            output_shape = [s for i, s in enumerate(input.shape) if i != dim]

        values = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        indices = torch.empty(output_shape, dtype=torch.long, device=input.device)
    else:
        values, indices = out

    # Flatten input tensor to 2D (rows x cols)
    n_rows = input.size(0)
    n_cols = input.size(dim)
    input_flat = input.transpose(0, dim).contiguous().view(n_rows, n_cols)

    # Launch the Triton kernel
    BLOCK_SIZE = 1024  # This should be tuned based on the hardware
    grid = (n_rows,)
    max_reduce_kernel[grid](input_flat, values, indices, input_flat.stride(0), n_cols, n_cols, BLOCK_SIZE=BLOCK_SIZE)

    if keepdim:
        values = values.unsqueeze(dim)
        indices = indices.unsqueeze(dim)

    return MaxResult(values, indices)
