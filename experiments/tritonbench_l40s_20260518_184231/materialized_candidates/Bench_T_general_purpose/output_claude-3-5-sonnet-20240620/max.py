import triton
import triton.language as tl

@triton.jit
def max_kernel(input_ptr, output_values_ptr, output_indices_ptr, dim, n_rows, n_cols):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    # Load the row from the input tensor
    row_data = tl.load(input_ptr + row * n_cols)

    # Initialize max value and index
    max_value = -tl.inf
    max_index = -1

    # Find max value and index
    for col in range(n_cols):
        if row_data[col] > max_value:
            max_value = row_data[col]
            max_index = col

    # Store results
    tl.store(output_values_ptr + row, max_value)
    tl.store(output_indices_ptr + row, max_index)

import torch
from collections import namedtuple

def max_wrapper(input: torch.Tensor, dim: int, keepdim: bool = False, out: tuple = None) -> tuple:
    n_rows, n_cols = input.shape
    output_values = torch.empty(n_rows, dtype=input.dtype, device=input.device)
    output_indices = torch.empty(n_rows, dtype=torch.long, device=input.device)

    # Launch the Triton kernel
    max_kernel[(n_rows,)](input, output_values, output_indices, dim, n_rows, n_cols)

    if keepdim:
        output_values = output_values.unsqueeze(dim)
        output_indices = output_indices.unsqueeze(dim)

    if out is not None:
        out[0].copy_(output_values)
        out[1].copy_(output_indices)
        return out
    return output_values, output_indices
