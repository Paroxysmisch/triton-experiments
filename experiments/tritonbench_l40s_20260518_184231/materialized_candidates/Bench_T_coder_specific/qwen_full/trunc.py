import torch
import triton

@triton.jit
def trunc(input):
    return input.to(torch.trunc(input).dtype)

def trunc(input, *, out=None):
    if out is None:
        return trunc(input)
    else:
        return trunc(input, out=out)
