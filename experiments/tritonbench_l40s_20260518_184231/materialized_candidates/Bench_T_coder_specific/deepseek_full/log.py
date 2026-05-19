import torch
import triton
import triton.language as tl

@triton.jit
def _log(x):
    # Math: y_{i} = \log_{e} (x_{i})
    return tl.math.log(x)

def log(input, *, out=None):
    # Type: (Tensor,)-> Tensor
    # Args: input (Tensor): the input tensor.
    # Returns: the function computes the natural logarithm (base e) of each element in the input tensor.
    # Example: torch.log(torch.tensor([1.0, 2.0, 3.0], device='triton'))
    # Result: tensor([0.0, 0.6931, 1.0986], device='triton')
    if input.is_floating_point():
        if out is None:
            return _log(input)
        else:
            _log(input, out=out)
            return out
    else:
        raise ValueError("log: expected a floating-point tensor")
