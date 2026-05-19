import torch
import triton
import triton.language as tl

@triton.jit
def min_func(x, y):
    return tl.minimum(x, y)

@triton.jit
def min_dim(x, dim):
    return tl.min(x, dim)

@triton.jit
def min_keepdim(x, dim, keepdim):
    return tl.min(x, dim, keepdim)

def min(input, dim, keepdim=False, *, out=None):
    if isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor):
        return min_func(input, other)
    elif isinstance(input, torch.Tensor):
        return min_dim(input, dim)
    elif isinstance(keepdim, bool):
        return min_keepdim(input, dim, keepdim)
    else:
        raise ValueError("Unsupported input types for min function")
