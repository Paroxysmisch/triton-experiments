import torch
import triton
import triton.language as tl

@triton.jit
def digamma_triton(x):
    # Convert input to float32
    x_fp32 = x.to(tl.float32)
    # Compute the logarithm of the absolute value of the input
    log_abs_x = tl.log(tl.abs(x_fp32))
    # Compute the psi function using a series expansion
    z = log_abs_x + 0.5
    s = tl.where(x_fp32 > 0, log_abs_x, 0.)
    z = tl.where(x_fp32 > 0, z, 0.)
    s += tl.math.lgamma(z + 1.)
    s += (z - 0.5) * (1. + (1. + 4. * z) * tl.math.rsqrt(16. * z * z + 1.))
    s += tl.where(x_fp32 > 0, 0., -tl.math.log(-x_fp32) - 1.44269504))
    return s

def digamma(input, *, out=None):
    return digamma_triton(input)
