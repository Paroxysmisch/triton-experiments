import torch
import triton
import triton.language as tl

@triton.jit
def i0_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    xx = x * x
    half_sq = 0.25 * xx

    # Approximate I0(x) via its series expansion
    sum_ = 1.0
    term = 1.0
    TERMS = 25
    for k in range(1, TERMS):
        term *= half_sq / (k * k)
        sum_ += term

    tl.store(out_ptr + offsets, sum_, mask=mask)

def i0(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    n = input.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    i0_kernel[grid](input, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out
