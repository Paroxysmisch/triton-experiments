import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    a_val = tl.load(A + offs, mask=mask, other=0.0)
    b_val = tl.load(B + offs, mask=mask, other=0.0)
    tl.store(C + offs, a_val + b_val, mask=mask)

def custom_add(a, b):
    c = torch.empty_like(a)
    size = a.numel()
    BLOCK = 16
    grid = lambda meta: (triton.cdiv(size, BLOCK),)
    _add_kernel[grid](a, b, c, size, BLOCK=BLOCK)
    return c
