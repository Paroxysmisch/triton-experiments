import triton
import torch

@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    prog_id = tl.program_id(0)
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    c = a + b
    tl.store(C + offs, c, mask=mask)

def custom_add(a, b):
    c = torch.empty_like(a)
    size = a.numel()
    BLOCK = 16
    grid = lambda opt: (triton.cdiv(size, BLOCK),)
    _add_kernel[grid](a, b, c, size, BLOCK=BLOCK)
    return c

# Example usage
a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')
result = custom_add(a, b)
