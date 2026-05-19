import triton
import triton.language as tl

@triton.jit
def tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_block = tl.load(input_ptr + offsets, mask=mask)
    output_block = tl.math.tanh(input_block)
    tl.store(output_ptr + offsets, output_block, mask=mask)

import torch
import triton
import triton.language as tl

def tanh(input, *, out=None):
    # Ensure the input is a tensor
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
    grid = (triton.cdiv(input.numel(), 1024),)
    tanh_kernel[grid](input, out, input.numel(), BLOCK_SIZE=1024)

    return out
