import triton
import triton.language as tl

@triton.jit
def signbit_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vec = tl.load(input_ptr + offsets, mask=mask)
    output_vec = input_vec < 0
    tl.store(output_ptr + offsets, output_vec, mask=mask)

import torch

def signbit(input, *, out=None):
    # Ensure input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    # Ensure input is a floating-point type
    if input.dtype not in [torch.float16, torch.float32, torch.float64]:
        raise TypeError("input must be a floating-point tensor")

    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input tensor")
        if out.dtype != torch.bool:
            raise TypeError("out tensor must be of type torch.bool")

    # Launch the Triton kernel
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    signbit_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)

    return out
