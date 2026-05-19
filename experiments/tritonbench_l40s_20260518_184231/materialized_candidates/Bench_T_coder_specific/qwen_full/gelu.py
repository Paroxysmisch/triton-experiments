import math
import torch
import triton
import triton.language as tl

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def gelu_approximation(x):
    # 0.044715 = sqrt(2 / pi)
    return 0.5 * x * (1 + tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x))

@triton.jit
def gelu_exact(x):
    return x * 0.5 * (1.0 + tl.math.erf(x / math.sqrt(2.0)))

def gelu(input, approximate='none'):
    if approximate == 'tanh':
        return gelu_approximation(input)
    elif approximate == 'none':
        return gelu_exact(input)
    else:
        raise ValueError(f"Invalid value for approximate: {approximate}")
