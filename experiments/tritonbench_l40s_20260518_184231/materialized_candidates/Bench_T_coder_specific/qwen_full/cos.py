import torch
import triton
import triton.language as tl

@triton.jit
def cos(input):
    return tl.cos(input)

def torch_cos(input, *, out=None):
    if out is None:
        return cos(input)
    else:
        return cos(input, out=out)
