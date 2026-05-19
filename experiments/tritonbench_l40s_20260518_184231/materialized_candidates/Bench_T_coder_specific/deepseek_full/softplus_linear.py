import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def softplus_linear(x, weight, bias, beta, threshold):
    # Apply linear transformation
    out = tl.dot(x, weight)
    
    # Apply bias if provided
    if bias is not None:
        out += bias
    
    # Apply softplus activation function
    out = tl.where(out <= threshold, tl.math.log(tl.math.exp(beta * out) + 1) / beta, out)
    
    return out

def softplus_linear_wrapper(input: Tensor, weight: Tensor, bias: Tensor = None, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    # Check if input tensor has the correct dimension
    if input.ndim != 2:
        raise ValueError(f"Expected input to have 2 dimensions, but got {input.ndim}")
    
    # Check if weight tensor has the correct dimension
    if weight.ndim != 2:
        raise ValueError(f"Expected weight to have 2 dimensions, but got {weight.ndim}")
    
    # Check if input and weight tensors have compatible dimensions
    if input.shape[1] != weight.shape[1]:
        raise ValueError(f"Incompatible dimensions between input and weight")
    
    # Check if bias tensor is provided and has the correct dimension
    if bias is not None and bias.shape[0] != weight.shape[0]:
        raise ValueError(f"Incompatible dimensions between weight and bias")
    
    # Call the Triton kernel
    output = softplus_linear(input, weight, bias, beta, threshold)
    
    return output
