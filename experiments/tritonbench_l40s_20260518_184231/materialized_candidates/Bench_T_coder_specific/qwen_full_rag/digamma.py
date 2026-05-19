import logging
import triton
import triton.language as tl
from .math import lgamma


# Digamma kernel
@triton.jit
def digamma_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_minus_1 = x - 1.0
    term1 = tl.where(
        x != 1.0,
        tl.math.rsqrt((x_minus_1) * (x_minus_1 + 1.0)) / (x_minus_1),
        0.5,
    )
    term2 = lgamma(x).to(tl.float32) / x
    y = term1 - term2
    tl.store(y_ptr + offsets, y.to(x_ptr.dtype.element_ty), mask=mask)


# Wrapper function for calling the above kernel
def digamma(input, *, out=None):
    logging.debug("GEMS DIGAMMA")
    assert input.is_contiguous()
    assert input.dim() >= 1
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape
        assert out.is_contiguous()
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    digamma_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
