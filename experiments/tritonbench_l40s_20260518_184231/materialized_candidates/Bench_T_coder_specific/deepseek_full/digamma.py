import torch
import triton
import triton.language as tl

# Triton kernel for computing the digamma function
@triton.jit
def triton_digamma(x, **mask):
    # Compute the digamma function using Triton
    # The digamma function is the derivative of the logarithm of the gamma function
    return tl.digamma(x, **mask)

# Wrapper function for calling the Triton kernel
def digamma(input, *, out=None):
    # Check if input is a tensor
    if not isinstance(input, torch.Tensor):
        # Convert input to a tensor
        input = torch.tensor(input)
    # Check if out is provided
    if out is not None:
        # If out is provided, use it as the output tensor
        return triton_digamma(input, **out)
    else:
        # If out is not provided, compute the digamma function directly
        return triton_digamma(input)
