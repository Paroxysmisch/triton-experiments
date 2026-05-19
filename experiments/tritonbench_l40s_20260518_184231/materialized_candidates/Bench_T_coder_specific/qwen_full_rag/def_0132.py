import triton
import triton.language as tl
import logging

@triton.jit
def _mul_sub_kernel(
    x_ptr,
    y_mul_ptr,
    y_sub_ptr,
    alpha,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y_mul = tl.load(y_mul_ptr + offsets, mask=mask)
    y_sub = tl.load(y_sub_ptr + offsets, mask=mask)
    
    out = x * y_mul - alpha * y_sub
    tl.store(out_ptr + offsets, out, mask=mask)


def mul_sub(x, y_mul, y_sub, alpha=1, out=None):
    logging.debug("GEMS MUL-SUB")
    assert x.shape == y_mul.shape == y_sub.shape
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    if out is None:
        out = torch.empty_like(x)
    _mul_sub_kernel[grid](x, y_mul, y_sub, alpha, out, n_elements, BLOCK_SIZE=1024)
    return out
