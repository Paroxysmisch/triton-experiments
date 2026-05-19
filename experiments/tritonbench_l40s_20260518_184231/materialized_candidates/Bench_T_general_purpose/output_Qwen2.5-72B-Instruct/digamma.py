import triton
import triton.language as tl

@triton.jit
def digamma_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    
    # Compute digamma using an approximation
    # This is a simplified version of the digamma function
    # For a more accurate implementation, you might want to use a more sophisticated approximation
    y = tl.where(x == 0.0, -float('inf'), tl.digamma(x))
    
    tl.store(Y + offsets, y, mask=mask)

import torch
import triton
import triton.language as tl

def digamma(input, *, out=None):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a Tensor")
    
    # Ensure input is on the same device
    device = input.device
    dtype = input.dtype
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input, device=device, dtype=dtype)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a Tensor")
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input")
        if out.device != device:
            raise ValueError("out tensor must be on the same device as input")
        if out.dtype != dtype:
            raise ValueError("out tensor must have the same dtype as input")
    
    # Launch the kernel
    grid = (triton.cdiv(input.numel(), 1024),)
    digamma_kernel[grid](input, out, input.numel(), BLOCK_SIZE=1024)
    
    return out

import torch

# Create a tensor
input = torch.tensor([1.0, 2.0, 0.0, 3.0], device='cuda')

# Compute digamma
output = digamma(input)

print(output)  # Output should be: tensor([ -0.5772,  0.4228, -inf,  0.9228], device='cuda')
