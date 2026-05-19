import triton
import triton.language as tl
from triton import Tensor

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def gelu_approx(x):
    # Approximate GELU uses custom tanh function
    return 0.5 * x * (1 + tanh(x * 0.79788456 * (1 + 0.044715 * x * x)))

@triton.jit
def gelu(x, approximate):
    # GELU activations function
    if approximate == 'tanh':
        return gelu_approx(x)
    else:
        return x * tl.sigmoid(1.702 * x)

def wrapper_gelu(input, approximate='none'):
    # Triton wrapper for GELU activations
    return gelu(input, approximate)
