import triton
import triton.language as tl

@triton.jit
def exp_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vec = tl.load(input_ptr + offsets, mask=mask)
    output_vec = tl.exp(input_vec)
    tl.store(output_ptr + offsets, output_vec, mask=mask)

import torch
import triton
import triton.language as tl

def exp(input, *, out=None):
    # Ensure input is a Torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input, device=input.device)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input tensor")
        if out.device != input.device:
            raise ValueError("out tensor must be on the same device as input tensor")

    # Launch the Triton kernel
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    exp_kernel[grid](
        input.contiguous().data_ptr(),
        out.contiguous().data_ptr(),
        n_elements,
        BLOCK_SIZE=1024,
    )

    return out
