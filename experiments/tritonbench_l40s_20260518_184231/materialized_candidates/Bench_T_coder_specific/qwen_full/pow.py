import logging
import torch
import triton
import triton.language as tl

@triton.jit
def pow_kernel(x, y, EXP):
    return tl.libdevice.pow(x, y) if isinstance(EXP, tl.tensor) else tl.libdevice.pow(x, EXP)

def pow(input, exponent, *, out=None):
    logging.debug("GEMS POW")
    if out == None:
        if isinstance(exponent, torch.Tensor):
            assert exponent.is_floating_point()
            exponent = exponent.to(input.dtype)
            out = pow_kernel(input, exponent, EXP=exponent)
        else:
            assert isinstance(exponent, float)
            out = pow_kernel(input, exponent, EXP=exponent)
    else:
        if isinstance(exponent, torch.Tensor):
            assert exponent.is_floating_point()
            exponent = exponent.to(input.dtype)
            pow_kernel(input, exponent, EXP=exponent, out=out)
        else:
            assert isinstance(exponent, float)
            pow_kernel(input, exponent, EXP=exponent, out=out)
    return out
