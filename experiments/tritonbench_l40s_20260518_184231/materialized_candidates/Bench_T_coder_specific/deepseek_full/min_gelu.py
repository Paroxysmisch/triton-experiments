import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Union
from enum import Enum

class Approximate(Enum):
    NONE = 'none'
    TANH = 'tanh'

@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def gelu_exact(input):
    """
    Gaussian Error Linear Unit (GELU)
    """
    pi = tl.constant(3.1415926535897932)
    coeff = tl.math.rsqrt(2.0 / pi)
    poly = 0.044715 * input * input * input
    return input * tl.math.sigmoid(input * coeff * (1 + poly))

@triton.jit
def gelu_approx(input):
    coeff = tl.math.rsqrt(2.0)
    poly = 0.044715 * input * input * input
    return input * 0.5 * (1 + tanh(coeff * (input + 0.044715 * input * input * input)))

@triton.jit
def minimum(a, b):
    return a if a < b else b

@triton.jit
def minimum_all(input):
    return tl.minimum(input)

@triton.jit
def minimum_dim(input, dim):
    return tl.minimum(input, dim=dim)

@triton.jit
def minimum_dim_keep(input, dim):
    return tl.minimum(input, dim=dim, keepdim=True)

def min_gelu(
    input: Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    approximate: str = 'none',
    out: Optional[Tensor] = None,
) -> Tensor:
    if approximate == Approximate.NONE.value:
        gelu = gelu_exact
    elif approximate == Approximate.TANH.value:
        gelu = gelu_approx
    else:
        gelu = gelu_exact

    if out is None:
        if dim is None:
            return minimum_all(gelu(input))
        elif isinstance(dim, int):
            if keepdim:
                return minimum_dim_keep(gelu(input), dim)
            else:
                return minimum_dim(gelu(input), dim)
        else:
            values = [minimum_dim_keep(gelu(input), d) for d in dim]
            return namedtuple("values", "values indices")(values, indices)
    else:
        gelu(input, out=out)
        if dim is None:
            return minimum_all(out)
        elif isinstance(dim, int):
            if keepdim:
                return minimum_dim_keep(out, dim)
            else:
                return minimum_dim(out, dim)
        else:
            values = [minimum_dim_keep(out, d) for d in dim]
            return namedtuple("values", "values indices")(values, indices)
