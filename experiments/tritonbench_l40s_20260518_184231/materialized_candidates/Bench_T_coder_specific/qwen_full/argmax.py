import torch
import triton
import triton.language as tl

@triton.jit
def argmax(input, dim, keepdim=False):
    return tl.argmax(input, axis=dim, keepdim=keepdim)

def argmax(input, dim, keepdim=False):
    if dim is None:
        input = input.flatten()
        dim = 0
    return argmax(input, dim, keepdim)
