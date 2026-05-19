import torch
import triton
import triton.language as tl

@triton.jit
def bessel_j1_kernel(x, *, out=None):
    # Convert input to float32
    x = x.to(tl.float32)
    # Compute Bessel function of the first kind of order 1
    return tl.math.j1(x)

def bessel_j1(input, *, out=None):
    # Wrapper function for Bessel function of the first kind of order 1
    if out is None:
        return bessel_j1_kernel(input)
    else:
        return bessel_j1_kernel(input, out=out)
