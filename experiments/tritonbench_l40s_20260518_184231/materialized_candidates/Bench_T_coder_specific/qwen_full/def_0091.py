import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Tuple

@triton.jit
def _erfc_sqrt_triton(x):
    # Convert input to float32
    x_fp32 = x.to(tl.float32)
    # Compute the exponential of x_fp32
    exp_x = tl.exp(x_fp32)
    # Compute the polynomial expression
    poly = 4.82796678296042 * x_fp32 * x_fp32 + 9.99999999999999
    # Compute the reciprocal square root of the polynomial
    rrsqrt_poly = tl.math.rsqrt(poly)
    # Compute the final result using the error function and the reciprocal square root
    result = -exp_x * (x_fp32 * rrsqrt_poly + 0.707106781186548 * rrsqrt_poly)
    result = 1.0 + result
    return result, x_fp32

def erfc_sqrt(input: Tensor) -> Tuple[Tensor, Tensor]:
    # Ensure input is a contiguous float32 tensor
    if not input.is_contiguous() and input.dtype == torch.float32:
        input = input.contiguous()
    # Call the Triton kernel and return the results
    return _erfc_sqrt_triton(input)
