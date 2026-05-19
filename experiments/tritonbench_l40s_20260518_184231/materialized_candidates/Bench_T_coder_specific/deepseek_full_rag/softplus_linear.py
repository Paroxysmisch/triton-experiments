import triton
import triton.language as tl
import torch

@triton.jit
def softplus(x, beta=1.0):
    """
    Softplus activation function.
    """
    return tl.log(tl.exp(beta * x) + 1) / beta

@triton.jit
def linear(input, weight, bias=None, beta=1):
    """
    Linear transformation.
    """
    output = tl.dot(input, weight)
    if beta != 1:
        output = output * beta
    if bias is not None:
        output = output + bias
    return output

@triton.jit
def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    """
    Applies a linear transformation to the input tensor, followed by the Softplus activation function applied element-wise.
    """
    output = linear(input, weight, bias, beta)
    output = softplus(output, beta)
    return output

def wrapper_softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    """
    Applies a linear transformation to the input tensor, followed by the Softplus activation function applied element-wise.
    """
    output = softplus_linear(input, weight, bias, beta, threshold)
    return output

# Example Usage
input = torch.tensor([1.0, 2.0, 3.0], device='cuda')
weight = torch.tensor([4.0, 5.0, 6.0], device='cuda')
bias = torch.tensor([7.0, 8.0, 9.0], device='cuda')
output = wrapper_softplus_linear(input, weight, bias, beta=2, threshold=20)
