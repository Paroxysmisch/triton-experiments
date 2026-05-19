import triton
import numpy as np

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    if dtype is not None:
        input = input.to(dtype)
        other = other.to(dtype)

    other = other * alpha
    result = input + other

    if dim is not None:
        result = result.mean(dim=dim, keepdim=keepdim)
    else:
        result = result.mean()

    return result
