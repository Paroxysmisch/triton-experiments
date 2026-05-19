import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def gelu_approx(x):
    return 0.5 * x * (1 + tanh(tl.sqrt(2 / 3.14159265) * (x + 0.044715 * x * x * x)))

@triton.jit
def gelu(x):
    return x * tl.sigmoid(1.702 * x)

def sub_gelu(input, other, alpha=1, approximate='none', out=None) -> Tensor:
    if approximate == 'none':
        return gelu(input - alpha * other)
    elif approximate == 'tanh':
        return gelu_approx(input - alpha * other)
    else:
        raise ValueError("Invalid approximate value, expected 'none' or 'tanh'")
