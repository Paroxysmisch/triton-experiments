import triton
import triton.language as tl

@triton.jit
def exp_sqrt_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr  # Block size
):
    # Compute the index of the current thread
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the input data
    input_data = tl.load(input_ptr + offsets, mask=mask)

    # Compute exp and sqrt
    output_data = tl.sqrt(tl.exp(input_data))

    # Store the result
    tl.store(output_ptr + offsets, output_data, mask=mask)

import torch
import triton
import triton.language as tl

def exp_sqrt(input, out=None) -> torch.Tensor:
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
    exp_sqrt_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)

    return out
