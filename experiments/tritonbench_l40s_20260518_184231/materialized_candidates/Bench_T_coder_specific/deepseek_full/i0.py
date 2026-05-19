import torch
import triton
import triton.language as tl
from . import utils

@triton.jit
def i0_kernel(x, out):
    # Compute the zeroth order modified Bessel function of the first kind
    x_square = x * x
    sum = tl.zeros([1], dtype=tl.float32)
    k = 0
    k_limit = 100
    while k < k_limit:
        term = (x_square / 4) ** k / utils.double_factorial(k)
        if k % 2 == 0:
            sum += term
        else:
            sum -= term
        k += 1
    i0 = sum
    tl.store(out, i0)

def i0(input, *, out=None):
    # Compute the zeroth order modified Bessel function of the first kind for each element of the input tensor
    if out is None:
        out = torch.empty_like(input)
    assert input.is_cuda and out.is_cuda
    n_elements = out.numel()
    grid = lambda meta: (n_elements,)
    i0_kernel[grid](input, out)
    return out
