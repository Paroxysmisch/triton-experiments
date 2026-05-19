import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the input tensor
    stride_dim,  # Stride along the specified dimension
    stride_other,  # Stride along the other dimensions
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vec = tl.load(input_ptr + offsets, mask=mask)
    max_val = tl.max(input_vec, axis=0)
    input_vec = input_vec - max_val

    exp_vec = tl.exp(input_vec)
    sum_exp = tl.sum(exp_vec, axis=0)

    output_vec = exp_vec / sum_exp
    tl.store(output_ptr + offsets, output_vec, mask=mask)

import torch
import triton
import triton.language as tl

def softmax(input, dim, dtype=None) -> torch.Tensor:
    if dtype is not None:
        input = input.to(dtype)

    # Get the shape and strides of the input tensor
    shape = input.shape
    strides = input.stride()

    # Ensure the specified dimension is within bounds
    if dim < 0:
        dim = len(shape) + dim
    if dim < 0 or dim >= len(shape):
        raise ValueError("Dimension out of range")

    # Compute the number of elements in the input tensor
    n_elements = input.numel()

    # Compute the strides along the specified dimension and the other dimensions
    stride_dim = strides[dim]
    stride_other = strides[0] if dim > 0 else 1

    # Allocate the output tensor
    output = torch.empty_like(input)

    # Define the grid and block sizes
    grid = (n_elements // 1024 + 1,)
    block = (1024,)

    # Launch the Triton kernel
    softmax_kernel[grid, block](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        n_elements=n_elements,
        stride_dim=stride_dim,
        stride_other=stride_other,
        BLOCK_SIZE=1024
    )

    return output
