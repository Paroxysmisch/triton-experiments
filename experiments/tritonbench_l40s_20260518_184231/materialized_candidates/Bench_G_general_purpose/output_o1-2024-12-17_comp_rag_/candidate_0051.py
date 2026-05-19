import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    a_vals = tl.load(A + offs, mask=mask, other=0.0)
    b_vals = tl.load(B + offs, mask=mask, other=0.0)
    c_vals = a_vals + b_vals
    tl.store(C + offs, c_vals, mask=mask)

def custom_add(a, b):
    assert a.is_cuda and b.is_cuda, "Tensors must be on GPU"
    assert a.shape == b.shape, "Input shapes must match"
    c = torch.empty_like(a)
    size = a.numel()
    BLOCK = 16
    grid = (triton.cdiv(size, BLOCK), )
    _add_kernel[grid](a, b, c, size, BLOCK=BLOCK)
    return c
