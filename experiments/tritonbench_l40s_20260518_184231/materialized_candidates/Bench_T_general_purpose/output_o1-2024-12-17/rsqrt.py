import triton
import triton.language as tl
import torch

@triton.jit
def _rsqrt_kernel(
    input_ptr, 
    output_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    x_sqrt = tl.sqrt(x)
    rsqrt_val = 1.0 / x_sqrt
    rsqrt_val = tl.where(x < 0, float('nan'), rsqrt_val)
    tl.store(output_ptr + offsets, rsqrt_val, mask=mask)

def rsqrt(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    n = input.numel()
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _rsqrt_kernel[grid](input, out, n, BLOCK_SIZE=1024)
    return out
