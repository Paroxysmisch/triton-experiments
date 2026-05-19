import logging
import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.cuda.libdevice import sum as _sum
except ImportError:
    try:
        from triton.language.math import sum as _sum
    except ImportError:
        from triton.language.libdevice import sum as _sum

@triton.jit
def _sum(x):
    return x

def sum(input, dim, keepdim=False, *, dtype=None):
    logging.debug("GEMS SUM")
    if dtype is None:
        dtype = input.dtype
    if isinstance(dim, (list, tuple)):
        logging.debug("GEMS SUM LIST")
        out = input
        for d in dim:
            out = sum(out, d, keepdim=True)
        if not keepdim:
            out = out.squeeze(dim)
        return out
    else:
        logging.debug("GEMS SUM DIM")
        shape = list(input.shape)
        if dim is None:
            out = input.view(-1)
            return _sum(out, axis=0, dtype=dtype)
        else:
            keepdim = min(shape[dim], 1)
            n = input.sum(axis=dim, keepdim=True, dtype=dtype)
            if not keepdim:
                n = n.squeeze(dim)
            return n
