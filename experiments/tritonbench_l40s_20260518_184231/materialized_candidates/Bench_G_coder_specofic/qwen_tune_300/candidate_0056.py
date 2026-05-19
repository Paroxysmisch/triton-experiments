import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    # Triton kernel for element-wise addition
    prog_id = tl.program_id(0)
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    c = a + b
    tl.store(C + offs, c, mask=mask)

def custom_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Wrapper function to invoke the Triton kernel
    assert a.shape == b.shape
    c = torch.empty_like(a)
    size = a.numel()
    BLOCK = 16
    grid = lambda meta: (triton.cdiv(size, BLOCK),)
    _add_kernel[grid](a, b, c, size, BLOCK)
    return c
