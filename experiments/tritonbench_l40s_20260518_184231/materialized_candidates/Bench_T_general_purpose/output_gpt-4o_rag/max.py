import torch
import triton
import triton.language as tl
from collections import namedtuple

@triton.jit
def max_kernel(input_ptr, max_ptr, argmax_ptr, input_row_stride, max_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    
    # Initialize max and argmax
    max_val = row
    argmax = col_offsets

    # Perform reduction to find max and argmax
    for i in range(1, BLOCK_SIZE):
        current_val = tl.load(input_ptrs + i, mask=(col_offsets + i) < n_cols, other=-float('inf'))
        max_val = tl.where(current_val > max_val, current_val, max_val)
        argmax = tl.where(current_val > max_val, col_offsets + i, argmax)

    # Write back max and argmax to DRAM
    max_row_start_ptr = max_ptr + row_idx * max_row_stride
    argmax_row_start_ptr = argmax_ptr + row_idx * max_row_stride
    tl.store(max_row_start_ptr, max_val, mask=col_offsets < n_cols)
    tl.store(argmax_row_start_ptr, argmax, mask=col_offsets < n_cols)

def max(input, dim, keepdim=False, *, out=None):
    # Flatten the tensor except for the dimension to be reduced
    input_reshaped = input.transpose(dim, -1).contiguous()
    n_rows, n_cols = input_reshaped.shape[:-1], input_reshaped.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Allocate output tensors
    max_shape = list(input_reshaped.shape)
    max_shape[-1] = 1 if keepdim else n_cols
    max_tensor = torch.empty(max_shape, device=input.device, dtype=input.dtype)
    argmax_tensor = torch.empty(max_shape, device=input.device, dtype=torch.long)

    # Launch Triton kernel
    max_kernel[(n_rows,)](
        input_reshaped,
        max_tensor,
        argmax_tensor,
        input_reshaped.stride(0),
        max_tensor.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape output tensors according to keepdim
    if not keepdim:
        max_tensor = max_tensor.squeeze(-1)
        argmax_tensor = argmax_tensor.squeeze(-1)

    # Return namedtuple
    MaxResult = namedtuple('MaxResult', ['values', 'indices'])
    return MaxResult(values=max_tensor, indices=argmax_tensor)

# Example usage
input_tensor = torch.tensor([[1, 3, 2], [4, 6, 5]], device='cuda')
result = max(input_tensor, dim=1, keepdim=True)
print(result.values)
print(result.indices)
