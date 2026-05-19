import torch
import triton
import triton.language as tl

@triton.jit
def asin_kernel(x, out):
    return tl.math.asin(x.to(tl.float32)).to(out.dtype.element_ty)

def asin(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape and out.dtype == input.dtype
    if input.is_floating_point():
        asin_kernel(input, out)
    else:
        out.copy_(input.to(torch.float))
        asin_kernel(out, out)
    return out
