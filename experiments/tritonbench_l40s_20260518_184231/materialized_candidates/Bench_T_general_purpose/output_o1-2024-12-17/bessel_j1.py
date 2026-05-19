import triton
import triton.language as tl
import torch

@triton.jit
def _bessel_j1_kernel(
    input_ptr, 
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Series approximation for J1(x):
    # J1(x) ≈ x/2 - x^3/16 + x^5/384 - x^7/18432
    xx = x * x
    res = x * 0.5
    res = res - (x * xx) * (1.0 / 16.0)
    res = res + (x * xx * xx) * (1.0 / 384.0)
    res = res - (x * xx * xx * xx) * (1.0 / 18432.0)
    tl.store(output_ptr + offsets, res, mask=mask)

def bessel_j1(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    n_elements = input.numel()
    assert input.is_cuda, "input must be a CUDA tensor."
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _bessel_j1_kernel[grid](
        input, out, n_elements, BLOCK_SIZE
    )
    return out
