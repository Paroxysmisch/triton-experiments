import triton
import triton.language as tl
import logging

@triton.jit
def _sub(input, other, alpha):
    return input - alpha * other

def sub(input, other, *, alpha=1, out=None):
    logging.debug("GEMS SUB")
    return _sub(input, other, alpha)
