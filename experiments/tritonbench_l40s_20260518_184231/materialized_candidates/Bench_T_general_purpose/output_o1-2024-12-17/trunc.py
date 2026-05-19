import triton
import triton.language as tl

@triton.jit
def _trunc_kernel(
    in_ptr, 
    out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    truncated_x = tl.where(x >= 0, tl.floor(x), tl.ceil(x))
    tl.store(out_ptr + offsets, truncated_x, mask=mask)

def trunc(input, *, out=None):
    if input.dtype.is_integer():
        if out is None:
            # Return a copy for integer inputs
            return input.clone()
        else:
            out.copy_(input)
            return out
    n_elements = input.numel()
    if out is None:
        out = triton.zeros_like(input)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _trunc_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
