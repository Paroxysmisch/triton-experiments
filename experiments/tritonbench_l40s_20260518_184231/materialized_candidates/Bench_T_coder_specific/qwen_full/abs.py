import torch
import triton
import triton.language as tl

@triton.jit
def abs(input):
    return tl.abs(input)

def abs(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    return abs(input) if input.is_floating_point() else abs(input).to(input.dtype)
