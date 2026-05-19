import torch
import triton
import triton.language as tl

@triton.jit
def log1p(input):
    return tl.math.log(tl.math.exp(input) + 1)

def log1p(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    return log1p(input, out=out)
