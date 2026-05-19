import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _digamma_approx_f32(x):
    # Special cases
    is_zero = x == 0.0
    is_neg = x < 0.0
    # Accumulate shifts if x < 6
    acc = 0.0
    tmp = x
    # Move x up to >= 6 for approximation
    i = 0
    while i < 20:  # limit loop count to avoid infinite loops
        keep_shifting = tmp < 6.0
        if keep_shifting:
            acc -= 1.0 / tmp
            tmp = tmp + 1.0
        i += 1
    # Polynomial (asymptotic) approximation for digamma at large x
    r = 1.0 / tmp
    res = tl.log(tmp) - 0.5 * r
    r2 = r * r
    # Use a few series terms for approximation
    res = res - r2 * (1.0 / 12.0) + r2 * r2 * (1.0 / 120.0) - (r2 * r2 * r2) * (1.0 / 252.0)
    res = acc + res

    # Handle special cases
    neg_val = float('nan')
    zero_val = float('-inf')
    # For x < 0 => NaN, except for x == 0 => -Inf
    res = tl.where(is_zero, zero_val, res)
    res = tl.where(is_neg & ~is_zero, neg_val, res)
    return res

@triton.jit
def digamma_kernel(
    x_ptr, 
    y_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    result = _digamma_approx_f32(x)
    tl.store(y_ptr + offsets, result, mask=mask)

def digamma(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    n_elements = input.numel()
    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    digamma_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
