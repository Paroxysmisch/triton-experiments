import torch
import triton
import triton.language as tl
from triton.language.libdevice import gammaln as _gammaln

@triton.jit
def gammaln(x):
    # Triton kernel to compute the natural logarithm of the absolute value of the gamma function
    return _gammaln(tl.abs(x))

def gammaln(input, *, out=None):
    # Wrapper function for the Triton kernel
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32, device=input.device)
    assert input.is_contiguous()
    assert out.is_contiguous()
    assert input.device == out.device
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gammaln[grid](input, out=out)
    return out
