import triton
import triton.language as tl

@triton.jit
def min_kernel(
    input_ptr,  # Pointer to the input tensor
    min_ptr,    # Pointer to the output tensor for minimum values
    idx_ptr,    # Pointer to the output tensor for indices of minimum values
    stride,     # Stride of the input tensor in the specified dimension
    n_elements, # Number of elements in the specified dimension
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x_offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK_SIZE), BLOCK_SIZE)
    mask = x_offsets < n_elements

    # Load the input values
    input_values = tl.load(input_ptr + x_offsets, mask=mask, other=float('inf'))
    input_indices = x_offsets

    # Initialize the minimum value and index
    min_value = tl.full((1,), float('inf'), tl.float32)
    min_index = tl.full((1,), -1, tl.int32)

    # Find the minimum value and its index
    for i in range(0, BLOCK_SIZE, 32):
        current_values = input_values[i:i + 32]
        current_indices = input_indices[i:i + 32]
        current_min_value = tl.minimum(min_value, tl.min(current_values, axis=0))
        current_min_index = tl.where(tl.equal(current_min_value, current_values), current_indices, min_index)
        min_value = current_min_value
        min_index = current_min_index

    # Write the results back to the output tensors
    tl.store(min_ptr + pid, min_value)
    tl.store(idx_ptr + pid, min_index)

import torch
import triton
import triton.language as tl

def min(input, dim, keepdim=False, *, out=None):
    # Check input tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    # Check dimension
    if not (0 <= dim < input.dim()):
        raise ValueError(f"dim must be in the range [0, {input.dim() - 1}]")
    
    # Check keepdim
    if not isinstance(keepdim, bool):
        raise TypeError("keepdim must be a boolean")
    
    # Determine the output shape
    output_shape = list(input.shape)
    if keepdim:
        output_shape[dim] = 1
    else:
        output_shape.pop(dim)
    
    # Create output tensors
    if out is None:
        min_tensor = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        min_indices = torch.empty(output_shape, dtype=torch.int64, device=input.device)
    else:
        min_tensor, min_indices = out
        if min_tensor.shape != output_shape or min_indices.shape != output_shape:
            raise ValueError("out tensors must have the correct shape")
    
    # Launch the Triton kernel
    grid = (input.shape[dim],)
    stride = input.stride(dim)
    n_elements = input.shape[dim]
    BLOCK_SIZE = 1024  # Adjust block size as needed

    min_kernel[grid](
        input, min_tensor, min_indices, stride, n_elements, BLOCK_SIZE
    )
    
    return min_tensor, min_indices
