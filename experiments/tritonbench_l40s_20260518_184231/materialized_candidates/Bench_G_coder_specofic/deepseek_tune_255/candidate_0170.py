import torch
import triton
import triton.language as tl
from math import isqrt

@triton.jit
def cos_func(a, b, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < a.shape[0]
    a_value = tl.load(a + offset, mask=mask)
    b_value = tl.cos(a_value)
    tl.store(b + offset, b_value, mask=mask)

def cos(a: torch.Tensor):
    b = torch.empty_like(a)
    BLOCK_SIZE = isqrt(a.numel())
    grid = lambda meta: (triton.cdiv(a.numel(), meta['BLOCK_SIZE']),)
    cos_func[grid](a, b, BLOCK_SIZE=BLOCK_SIZE)
    return b

a = torch.randn(7, device='cuda')
b = cos(a)
print(b)
