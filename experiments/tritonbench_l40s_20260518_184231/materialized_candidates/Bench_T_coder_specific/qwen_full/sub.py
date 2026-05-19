import torch
import triton
import triton.language as tl

@triton.jit
def sub(input, other, alpha):
    return input - alpha * other

def sub(input, other, *, alpha=1, out=None):
    if out is None:
        return sub(input, other, alpha)
    else:
        return sub(input, other, alpha, out)
