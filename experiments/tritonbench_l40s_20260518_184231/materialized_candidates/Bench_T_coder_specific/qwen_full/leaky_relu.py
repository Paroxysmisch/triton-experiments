import torch
import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(x, negative_slope):
    # Convert input to float32
    x_fp32 = x.to(tl.float32)
    # Compute LeakyReLU
    return tl.where(x_fp32 >= 0, x_fp32, negative_slope * x_fp32)

def leaky_relu(input, negative_slope=0.01, inplace=False):
    if inplace:
        input.mul_(leaky_relu_kernel(input, negative_slope))
        return input
    else:
        return leaky_relu_kernel(input, negative_slope)
