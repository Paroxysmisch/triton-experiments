import torch
import triton
import triton.language as tl
from triton.language.libdevice import erf, exp, pow, tanh

@triton.jit
def tanh(x):
    """Tanh activation function"""
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def gelu(x, approximate='none'):
    """Gaussian Error Linear Units (GELU)"""
    if approximate == 'none':
        return x * 0.5 * (1.0 + erf(x / 1.41421356237))
    elif approximate == 'tanh':
        return 0.5 * x * (1 + tanh(tl.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * pow(x, 3)))
    else:
        raise ValueError(f"Invalid approximate value: {approximate}")

@triton.jit
def minimum(a, b):
    """Minimum value between a and b"""
    return tl.math.min(a, b)

@triton.jit
def min2(a, dim=None, keepdim=False):
    """Minimum value along specified dimension"""
    return tl.reduce(a, dim, minimum, keepdim)

def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    if dim is None:
        input = input.flatten()
        dim = 0
    out = min2(gelu(input, approximate), dim, keepdim)
    if isinstance(out, tuple):
        out = type(out)(gelu(out[0], approximate), out[1])
    return out
