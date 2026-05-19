import torch
import triton
import triton.language as tl

@triton.jit
def log(input):
    return tl.math.log(input)

def log(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_contiguous()
    if input.dtype.is_fp64():
        out = log(input, out=out)
    elif input.dtype.is_fp32():
        out = log(input.to(tl.float32), out=out).to(tl.float16)
    else:
        pass
    return out
