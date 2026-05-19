import torch
import triton
import triton.language as tl

@triton.jit
def max_kernel(
    values_ptr, indices_ptr, input_ptr, input_row_stride, n_cols, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    
    max_val = tl.max(row, axis=0)
    max_idx = tl.argmax(row, axis=0)
    
    values_row_start_ptr = values_ptr + row_idx * input_row_stride
    indices_row_start_ptr = indices_ptr + row_idx * input_row_stride
    
    values_output_ptr = values_row_start_ptr + col_offsets
    indices_output_ptr = indices_row_start_ptr + col_offsets
    
    tl.store(values_output_ptr, max_val, mask=col_offsets < 1)
    tl.store(indices_output_ptr, max_idx, mask=col_offsets < 1)

import torch
from collections import namedtuple

def max(input, dim, keepdim=False, *, out=None):
    if dim < 0:
        dim = input.dim() + dim
    
    n_rows = input.size(dim)
    n_cols = input.size(dim + 1) if dim + 1 < input.dim() else 1
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    if out is None:
        values_shape = list(input.shape)
        indices_shape = list(input.shape)
        if not keepdim:
            values_shape.pop(dim)
            indices_shape.pop(dim)
        else:
            values_shape[dim] = 1
            indices_shape[dim] = 1
        values = torch.empty(values_shape, device=input.device, dtype=input.dtype)
        indices = torch.empty(indices_shape, device=input.device, dtype=torch.long)
    else:
        values, indices = out
        assert values.shape == indices.shape, "Output tensors must have the same shape"
        assert values.device == input.device, "Output tensors must be on the same device as input"
    
    max_kernel[(n_rows,)](
        values, indices, input, input.stride(dim), n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return namedtuple('MaxResult', ['values', 'indices'])(values, indices)

# Test case 1: Basic usage
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
result = max(input_tensor, dim=1)
print(result.values)  # Expected: tensor([[3], [6]])
print(result.indices)  # Expected: tensor([[2], [2]])

# Test case 2: keepdim=True
result = max(input_tensor, dim=1, keepdim=True)
print(result.values)  # Expected: tensor([[3], [6]])
print(result.indices)  # Expected: tensor([[2], [2]])

# Test case 3: Negative dimension
result = max(input_tensor, dim=-1)
print(result.values)  # Expected: tensor([[3], [6]])
print(result.indices)  # Expected: tensor([[2], [2]])

# Test case 4: Using out parameter
out_values = torch.empty((2, 1), device='cuda', dtype=torch.float32)
out_indices = torch.empty((2, 1), device='cuda', dtype=torch.long)
result = max(input_tensor, dim=1, out=(out_values, out_indices))
print(result.values)  # Expected: tensor([[3], [6]])
print(result.indices)  # Expected: tensor([[2], [2]])
