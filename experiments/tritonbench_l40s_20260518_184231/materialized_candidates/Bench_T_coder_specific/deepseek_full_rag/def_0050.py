import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def sqrt_exp_kernel(input, out, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the offsets for each block
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask for boundary conditions
    mask = offsets < n_elements
    # Load the input elements with the mask
    x = tl.load(input + offsets, mask=mask)
    # Compute the square root of the input elements
    x = tl.sqrt(x)
    # Compute the exponential of the square root
    x = tl.exp(x)
    # Store the result in the output tensor with the mask
    tl.store(out + offsets, x, mask=mask)

def sqrt_exp(input: Tensor, out: Tensor = None) -> Tensor:
    if out is None:
        out = torch.empty_like(input)

    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sqrt_exp_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
