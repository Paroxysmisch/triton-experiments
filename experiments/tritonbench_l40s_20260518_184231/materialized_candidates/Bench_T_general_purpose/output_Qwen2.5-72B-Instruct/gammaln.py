import triton
import triton.language as tl

@triton.jit
def gammaln_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_values = tl.load(input_ptr + offsets, mask=mask)
    abs_input_values = tl.abs(input_values)
    output_values = tl.math.lgamma(abs_input_values)

    tl.store(output_ptr + offsets, output_values, mask=mask)

import torch
import triton
import triton.language as tl

def gammaln(input, *, out=None):
    # Ensure input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input tensor")

    # Launch the Triton kernel
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    gammaln_kernel[grid](
        input,  # Pointer to the input tensor
        out,  # Pointer to the output tensor
        n_elements,  # Number of elements in the tensor
        BLOCK_SIZE=1024,  # Block size
    )

    return out
