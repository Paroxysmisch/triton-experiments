import triton
import triton.language as tl
import logging

@triton.jit
def div_kernel(x_ptr, y_ptr, output_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x / y
    tl.store(output_ptr + offsets, output, mask=mask)


def div(x, y, *, rounding_mode=None, out=None):
    logging.debug("GEMS DIV")
    assert rounding_mode in [None, "floor", "trunc"], \
        "Only 'floor' and 'trunc' rounding modes are currently supported"
    if isinstance(y, int) or isinstance(y, float):
        assert y != 0, "Division by zero is not allowed"
    elif isinstance(y, torch.Tensor):
        assert y.ndim == 0, "Division by tensor of dimension != 0 is not yet supported"

    if out is None:
        out = torch.empty_like(x, dtype=x.dtype)
    assert out.is_contiguous()

    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    div_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out
