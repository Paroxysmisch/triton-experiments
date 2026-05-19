import triton
import triton.language as tl
import math

BLOCK_SIZE = 1024

@triton.jit
def _dropout(x_ptr, x_keep_ptr, output_ptr, n_elements, p, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < n_elements
    x = tl.load(x_ptr + block_start, mask=mask, other=0.0)
    keep = tl.load(x_keep_ptr + block_start, mask=mask, other=0.0)
    out = tl.where(keep > 0.0, x / (1 - p), 0.0)
    tl.store(output_ptr + block_start, out, mask=mask)

def dropout(x, x_keep, p):
    if not x.is_contiguous():
        x = x.contiguous()
    if not x_keep.is_contiguous():
        x_keep = x_keep.contiguous()
    n_elements = x.numel()
    grid = lambda meta: ( (n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    _dropout[grid](
        x, x_keep, x, n_elements, p,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return x
