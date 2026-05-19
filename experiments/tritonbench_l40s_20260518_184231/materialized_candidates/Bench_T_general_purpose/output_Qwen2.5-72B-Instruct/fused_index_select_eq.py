import triton
import triton.language as tl

@triton.jit
def fused_index_select_eq_kernel(
    input_ptr,  # Pointer to the input tensor
    dim,        # Dimension along which to index
    index_ptr,  # Pointer to the index tensor
    other_ptr,  # Pointer to the other tensor or scalar
    out_ptr,    # Pointer to the output tensor
    input_shape,  # Shape of the input tensor
    other_shape,  # Shape of the other tensor
    index_size,   # Size of the index tensor
    stride,      # Stride of the input tensor along the dimension
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < index_size

    # Load indices
    indices = tl.load(index_ptr + offsets, mask=mask)

    # Compute the linear indices for the input tensor
    linear_indices = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for d in range(len(input_shape)):
        if d == dim:
            linear_indices += indices * stride[d]
        else:
            linear_indices += tl.arange(0, input_shape[d]) * stride[d]

    # Load the selected elements from the input tensor
    selected_elements = tl.load(input_ptr + linear_indices, mask=mask)

    # Load the other tensor or scalar
    if other_shape == 1:
        other = tl.load(other_ptr)
    else:
        other_offsets = tl.arange(0, other_shape[0])
        other = tl.load(other_ptr + other_offsets, mask=mask)

    # Perform element-wise equality comparison
    result = selected_elements == other

    # Store the result in the output tensor
    tl.store(out_ptr + offsets, result, mask=mask)

import torch
import triton
import triton.language as tl

def fused_index_select_eq(input, dim, index, other, *, out=None):
    # Validate input shapes and types
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a Tensor")
    if not isinstance(dim, int):
        raise TypeError("dim must be an integer")
    if not isinstance(index, (torch.IntTensor, torch.LongTensor)):
        raise TypeError("index must be an IntTensor or LongTensor")
    if not isinstance(other, (torch.Tensor, float)):
        raise TypeError("other must be a Tensor or a float")

    # Determine the shape of the selected tensor
    input_shape = list(input.shape)
    selected_shape = input_shape.copy()
    selected_shape[dim] = index.shape[0]

    # Determine the shape of the other tensor
    if isinstance(other, float):
        other = torch.tensor(other, device=input.device, dtype=input.dtype)
    other_shape = list(other.shape)

    # Ensure shapes are broadcastable
    if len(other_shape) > 1:
        raise ValueError("other must be a scalar or a 1D tensor")
    if len(other_shape) == 1 and other_shape[0] != selected_shape[dim]:
        raise ValueError("other must be broadcastable to the selected tensor shape")

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty(selected_shape, device=input.device, dtype=torch.bool)

    # Compute strides for the input tensor
    stride = input.stride()

    # Launch the Triton kernel
    grid = (index.numel() + 1024 - 1) // 1024
    fused_index_select_eq_kernel[grid](
        input, dim, index, other, out,
        input_shape, other_shape, index.numel(), stride,
        BLOCK_SIZE=1024
    )

    return out
