import triton
import triton.language as tl

@triton.jit
def mul_relu_kernel(
    input_ptr,  # Pointer to the input tensor
    other_ptr,  # Pointer to the other tensor
    output_ptr, # Pointer to the output tensor
    n_elements, # Number of elements
    BLOCK_SIZE: tl.constexpr, # Block size
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_vals = tl.load(input_ptr + offsets, mask=mask)
    other_vals = tl.load(other_ptr + offsets, mask=mask)

    result = input_vals * other_vals
    result = tl.where(result > 0, result, 0)  # ReLU

    tl.store(output_ptr + offsets, result, mask=mask)

import torch
import triton
import triton.language as tl
from torch.nn.functional import relu

def mul_relu(input, other, inplace=False, out=None):
    # Ensure input and other are tensors
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)

    # Ensure input and other have the same shape
    if input.shape != other.shape:
        raise ValueError("input and other must have the same shape")

    # Determine the output tensor
    if out is None:
        if inplace:
            out = input
        else:
            out = torch.empty_like(input)
    else:
        if inplace and out is not input:
            raise ValueError("inplace=True but out is not input")
        if out.shape != input.shape:
            raise ValueError("out must have the same shape as input")

    # Launch the Triton kernel
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mul_relu_kernel[grid](
        input, other, out, n_elements, BLOCK_SIZE=1024
    )

    return out
