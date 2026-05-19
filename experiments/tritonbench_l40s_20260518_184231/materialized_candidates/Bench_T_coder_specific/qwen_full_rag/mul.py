import triton
import triton.language as tl
import logging

@triton.jit
def _mul_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)

def mul(x, y, *, out=None):
    logging.debug("GEMS MUL")
    if out is None:
        out = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and out.is_cuda
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _mul_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out
