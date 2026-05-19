import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(
    A, B, C, size,
    BLOCK: tl.constexpr,
):
    prog_id = tl.program_id(axis=0)
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    c = a + b
    tl.store(C + offs, c, mask=mask)

def custom_add(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape, "Input tensors must have the same shape"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU"
    c = torch.empty_like(a)
    size = a.numel()
    BLOCK = 16
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK']), )
    _add_kernel[grid](a, b, c, size, BLOCK=BLOCK)
    return c
