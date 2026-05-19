import triton
import torch as pt
import numpy as np

@triton.jit
def _add_kernel(A, B, C, size, BLOCK):
    offs = triton.program_id(0) * BLOCK + triton.arange(0, BLOCK)
    mask = offs < size
    x = triton.load(A + offs, mask=mask)
    y = triton.load(B + offs, mask=mask)
    z = x + y
    triton.store(C + offs, z, mask=mask)

def custom_add(a, b):
    c = pt.empty_like(a)
    size = a.numel()
    BLOCK = 16
    grid = triton.cdiv(size, BLOCK)
    _add_kernel[grid](a.data.ptr(), b.data.ptr(), c.data.ptr(), size, BLOCK)
    return c
