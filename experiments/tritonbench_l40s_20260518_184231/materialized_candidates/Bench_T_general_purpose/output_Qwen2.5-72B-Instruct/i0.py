import triton
import triton.language as tl

@triton.jit
def i0_kernel(X, Y, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    x = tl.load(X + offsets, mask=mask)
    
    # Compute the zeroth order modified Bessel function of the first kind
    y = tl.zeros_like(x)
    k = 0
    term = tl.ones_like(x)
    while tl.any(term > 1e-10):
        y += term
        k += 1
        term *= (x * x / 4) / (k * k)
    
    tl.store(Y + offsets, y, mask=mask)

import torch
import triton
import triton.language as tl

def i0(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure the input tensor is on the same device as the output tensor
    assert input.device == out.device, "Input and output tensors must be on the same device"
    
    # Get the size of the input tensor
    size = input.numel()
    
    # Define the grid and block sizes
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(size, BLOCK_SIZE),)
    
    # Launch the kernel
    i0_kernel[grid](input, out, size, BLOCK_SIZE)
    
    return out
