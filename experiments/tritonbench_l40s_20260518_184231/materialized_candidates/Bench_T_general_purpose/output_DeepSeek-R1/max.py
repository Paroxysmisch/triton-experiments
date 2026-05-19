import torch
import triton
import triton.language as tl

@triton.jit
def max_kernel(
    input_ptr,
    output_values_ptr,
    output_indices_ptr,
    row_stride,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * row_stride

    max_value = -tl.inf
    max_index = 0

    for k in range(0, dim_size):
        ptr = row_start + k
        current_val = tl.load(ptr)
        current_index = k
        if current_val > max_value or (current_val == max_value and current_index < max_index):
            max_value = current_val
            max_index = current_index

    tl.store(output_values_ptr + row_idx, max_value)
    tl.store(output_indices_ptr + row_idx, max_index)

def max(input, dim, keepdim=False, *, out=None):
    # Ensure input is contiguous
    input = input.contiguous()

    # Check dim is valid
    if dim < 0 or dim >= input.dim():
        raise ValueError("dim out of range")

    # Collapse all dimensions except dim into a single dimension
    dim_size = input.size(dim)
    outer_dims = input.shape[:dim] + input.shape[dim+1:]
    outer_dim = 1
    for d in outer_dims:
        outer_dim *= d

    # Reshape input to 2D: (outer_dim, dim_size)
    input_2d = input.view(outer_dim, dim_size)

    # Create output tensors
    output_values = torch.empty((outer_dim,), dtype=input.dtype, device=input.device)
    output_indices = torch.empty((outer_dim,), dtype=torch.int64, device=input.device)

    # Launch kernel
    grid = (outer_dim,)
    max_kernel[grid](
        input_2d,
        output_values,
        output_indices,
        input_2d.stride(0),
        dim_size,
        BLOCK_SIZE=1024,
    )

    # Reshape outputs to the desired shape
    output_shape = list(input.shape)
    if keepdim:
        output_shape[dim] = 1
    else:
        output_shape.pop(dim)
    output_values = output_values.view(output_shape)
    output_indices = output_indices.view(output_shape)

    # Handle out parameter
    if out is not None:
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("out must be a tuple of two tensors")
        out[0].copy_(output_values)
        out[1].copy_(output_indices)
        return out
    else:
        return (output_values, output_indices)
