import torch
import triton
import triton.language as tl

@triton.jit
def abs(x):
    # Math: y = |x|
    return tl.where(x >= 0, x, -x)

def wrapper_abs(input, *, out=None):
    # Type: torch.tensor
    # Args: input (torch.tensor): the input tensor.
    # Keyword args: out (torch.tensor, optional): the output tensor.
    # Returns: torch.tensor: the output tensor.
    return abs(input, out=out)
