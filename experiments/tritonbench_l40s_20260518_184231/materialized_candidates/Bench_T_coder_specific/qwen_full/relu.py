import torch
import triton
import triton.language as tl

@triton.jit
def relu(input):
    return tl.where(input > 0, input, 0)

def relu(input, inplace=False):
    if inplace:
        return relu_(input)
    else:
        return relu_(input.clone())
