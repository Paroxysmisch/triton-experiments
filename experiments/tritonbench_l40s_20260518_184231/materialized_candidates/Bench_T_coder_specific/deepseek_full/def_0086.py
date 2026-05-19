import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def log_tanh(input, out=None) -> Tensor:
    # Compute the natural logarithm of each element in the input tensor
    input = tl.math.log(input)
    # Apply the hyperbolic tangent function to the result
    input = tl.math.tanh(input)
    return input

def call_log_tanh(input, out=None):
    # Ensure all input elements are positive for the logarithm function
    assert all(x > 0 for x in input), "All input elements must be positive for the logarithm function"
    return log_tanh(input, out)
