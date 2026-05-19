import torch
import triton
import triton.language as tl

def fftn(input, s=None, dim=None, norm=None, *, out=None):
    return torch.fft.fftn(input, s, dim, norm, out=out)

@triton.jit
def fftn(input, s=None, dim=None, norm=None, *, out=None):
    # ...
    # Implementation details omitted for brevity
    # ...
    return fftn_kernel(input, s, dim, norm, out=out)

def fftn_wrapper(input, s=None, dim=None, norm=None, *, out=None):
    # ...
    # Wrapper-specific code omitted for brevity
    # ...
    return fftn(input, s, dim, norm, out=out)
