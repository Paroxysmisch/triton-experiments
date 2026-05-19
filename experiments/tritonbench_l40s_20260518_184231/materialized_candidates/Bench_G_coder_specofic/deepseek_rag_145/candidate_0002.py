import triton.language as tl
import torch

@triton.jit
def kldivergence_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    n_elements: tl.constexpr):

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x * tl.log(x / y)

    tl.store(output_ptr + offsets, output, mask=mask)

def kldivergence(x, y, out=None):
    assert x.device == y.device
    if out is None:
        out = torch.empty_like(x)
    assert out.device == x.device
    
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    kldivergence_kernel[grid](x, y, out, n_elements=n_elements)
    return out
