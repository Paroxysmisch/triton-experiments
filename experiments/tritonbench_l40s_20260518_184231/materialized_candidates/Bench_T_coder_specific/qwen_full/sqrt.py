import torch
import triton
import triton.language as tl

@triton.jit
def sqrt(input):
    return tl.math.sqrt(input)

def sqrt(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    if input.dtype.is_fp64():
        out = sqrt(input, out=out)
    else:
        # TODO: support int/uint types
        assert input.dtype.is_fp64(), "only fp64 is supported currently"
    return out
