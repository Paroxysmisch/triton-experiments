import triton
import torch

@triton.jit
def tanh(input, *, out=None):
    # Calculate the hyperbolic tangent of the input
    output = torch.tanh(input)
    return output

def tanh_wrapper(input, *, out=None):
    # Call the Triton kernel to calculate the hyperbolic tangent
    return tanh(input, out=out)
