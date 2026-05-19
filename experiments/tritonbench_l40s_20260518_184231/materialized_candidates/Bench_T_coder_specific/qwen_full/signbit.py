import logging
import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.cuda.libdevice import signbit as _signbit
except ImportError:
    try:
        from triton.language.math import signbit as _signbit
    except ImportError:
        from triton.language.libdevice import signbit as _signbit

@triton.jit
def signbit(input):
    return _signbit(input)

def signbit(input, *, out=None):
    logging.debug("GEMS SIGNBIT")
    if out == None:
        return signbit(input)
    else:
        return signbit(input, out=out)
