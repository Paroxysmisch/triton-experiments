import torch
import triton
import triton.language as tl

@triton.jit
def _gammaln_kernel(
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
    x_abs = tl.abs(x)

    # Small epsilon for domain safety
    EPS = 1e-7
    x_abs = tl.where(x_abs == 0, EPS, x_abs)

    # Stirling's approximation for ln(Gamma(x)) ~ (x - 0.5)*ln(x) - x + 0.5*ln(2*pi)
    lnsqrt2pi = 0.9189385332046727
    result = (x_abs - 0.5) * tl.log(x_abs) - x_abs + lnsqrt2pi

    tl.store(output_ptr + offsets, result, mask=mask)

def gammaln(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _gammaln_kernel[grid](
        input,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
