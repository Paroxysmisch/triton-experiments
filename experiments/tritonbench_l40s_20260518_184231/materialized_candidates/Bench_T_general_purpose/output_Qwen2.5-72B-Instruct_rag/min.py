import torch
import triton
import triton.language as tl

@triton.jit
def min_kernel(
    min_values_ptr, min_indices_ptr, input_ptr, input_row_stride, n_cols, BLOCK_SIZE: tl.constexpr
):
    # The rows are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=float('inf'))
    
    # Initialize min_value and min_index
    min_value = tl.full((1,), float('inf'), dtype=tl.float32)
    min_index = tl.full((1,), -1, dtype=tl.int32)
    
    # Find the minimum value and its index
    for i in range(n_cols):
        value = row[i]
        if value < min_value:
            min_value = value
            min_index = i
    
    # Write back the minimum value and index to DRAM
    tl.store(min_values_ptr + row_idx, min_value)
    tl.store(min_indices_ptr + row_idx, min_index)

import torch

def min(input, dim, keepdim=False, *, out=None):
    # Validate input dimensions
    if dim < 0 or dim >= input.dim():
        raise ValueError(f"dim {dim} is out of bounds for input of size {input.size()}")
    
    # Determine the shape of the output tensors
    output_shape = list(input.size())
    if not keepdim:
        output_shape[dim] = 1
    else:
        output_shape[dim] = 1
    
    # Allocate output tensors
    if out is None:
        min_values = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        min_indices = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    else:
        min_values, min_indices = out
        if min_values.shape != output_shape or min_indices.shape != output_shape:
            raise ValueError(f"Output tensors must have shape {output_shape}")
    
    # Flatten the input tensor along the specified dimension
    input_reshaped = input.view(-1, input.size(dim))
    n_rows, n_cols = input_reshaped.shape
    
    # The block size is the smallest power of two greater than the number of columns in `input_reshaped`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Enqueue the kernel
    min_kernel[(n_rows,)](
        min_values.view(-1),
        min_indices.view(-1),
        input_reshaped.view(-1),
        input_reshaped.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape the output tensors if keepdim is False
    if not keepdim:
        min_values = min_values.squeeze(dim)
        min_indices = min_indices.squeeze(dim)
    
    return min_values, min_indices
