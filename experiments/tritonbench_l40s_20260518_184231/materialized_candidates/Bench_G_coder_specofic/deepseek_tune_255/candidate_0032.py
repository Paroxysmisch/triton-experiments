import torch
import triton
import triton.language as tl

@triton.jit
def _dropout(
    x_ptr,
    x_keep_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def dropout(x, x_keep, p):
    assert x.is_contiguous()
    assert x_keep.is_contiguous()
    num_elements = x.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']), )
    _dropout[grid](x, x_keep, x_keep, num_elements, BLOCK_SIZE=1024)
    return x_keep
