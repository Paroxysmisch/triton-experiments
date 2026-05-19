import triton
import triton.language as tl
import math

@triton.jit
def _asin_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    # Check range and set out of range values to NaN
    out_of_range = (x < -1.0) | (x > 1.0)
    x = tl.where(out_of_range, float('nan'), x)
    # Compute arcsine using libdevice
    x = tl.libdevice.asin(x)
    tl.store(out_ptr + offsets, x, mask=mask)

def asin(input, *, out=None):
    if out is None:
        out = triton.zeros_like(input)

    n_elements = input.numel
    BLOCK_SIZE = 1024
    grid = lambda meta: ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    _asin_kernel[grid](input, out, n_elements, BLOCK_SIZE)
    return out
