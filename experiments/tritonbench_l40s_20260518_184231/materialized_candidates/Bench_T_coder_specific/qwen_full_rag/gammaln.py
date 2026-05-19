import torch
import triton
import triton.language as tl
from triton.language.libdevice import lgamma

@triton.jit
def gammaln_kernel(input, output, n_elements,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input + offsets, mask=mask)
    output_x = lgamma(tl.abs(x))
    tl.store(output + offsets, output_x, mask=mask)

def gammaln(input, *, out=None):
    assert input.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    assert input.is_contiguous()
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_contiguous()
        assert out.shape == input.shape
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gammaln_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
