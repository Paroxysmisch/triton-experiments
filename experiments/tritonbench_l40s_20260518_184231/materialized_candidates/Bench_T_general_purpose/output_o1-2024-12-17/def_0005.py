import triton
import triton.language as tl

@triton.jit
def _relu_sqrt_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)
    x = tl.sqrt(x)
    tl.store(out_ptr + offsets, x, mask=mask)

def relu_sqrt(input, inplace=False, out=None):
    import math
    if inplace and out is not None:
        raise ValueError("Cannot use both inplace=True and out.")
    if out is None:
        if inplace:
            out = input
        else:
            out = input.clone()
    n_elements = input.numel()
    grid = lambda meta: (math.ceil(n_elements / meta['BLOCK_SIZE']),)
    _relu_sqrt_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
