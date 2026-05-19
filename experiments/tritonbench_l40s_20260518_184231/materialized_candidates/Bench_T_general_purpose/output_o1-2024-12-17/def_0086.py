import triton
import triton.language as tl
import torch
import math

@triton.jit
def _log_tanh_kernel(
    in_ptr, 
    out_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)
    x = tl.log(x)
    x = tl.tanh(x)
    tl.store(out_ptr + offsets, x, mask=mask)

def log_tanh(input, out=None):
    if not torch.all(input > 0):
        raise ValueError("All elements of 'input' must be positive for log_tanh.")

    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (math.ceil(n_elements / meta['BLOCK_SIZE']),)

    if out is None:
        out = torch.empty_like(input)

    _log_tanh_kernel[grid](
        input, 
        out, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
