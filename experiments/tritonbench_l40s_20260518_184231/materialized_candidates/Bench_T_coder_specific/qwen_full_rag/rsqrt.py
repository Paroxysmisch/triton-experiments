import torch
import triton
import triton.language as tl

@triton.jit
def rsqrt_kernel(
        x_ptr,
        y_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = 1 / tl.sqrt(x)
    tl.store(y_ptr + offsets, y, mask=mask)

def rsqrt(x, *, out=None):
    if out is None:
        out = torch.empty_like(x)
    else:
        assert x.is_contiguous()
    
    assert out.is_contiguous()
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    rsqrt_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out
