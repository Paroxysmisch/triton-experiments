import triton
import torch

@triton.jit
def bessel_j1_kernel(x, out):
    # Compute the Bessel function of the first kind of order 1 for each element
    return torch.bessel_j1(x)

def bessel_j1(input, *, out=None):
    # Check if the input tensor is on CUDA device
    if not input.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    
    # If an output tensor is not provided, create a new tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Launch the Triton kernel
    bessel_j1_kernel[(input.numel(),)](input, out)
    
    return out
