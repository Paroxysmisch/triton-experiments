import torch
import triton
import triton.language as tl
from typing import Optional

device = 'cuda:0'

@triton.jit
def i0_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    K_MAX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    x_squared = x * x
    x_squared_over_4 = x_squared / 4.0

    current_term = 1.0
    bessel_sum = current_term

    for k in range(1, K_MAX + 1):
        current_term *= x_squared_over_4 / (k * k)
        bessel_sum += current_term

    tl.store(output_ptr + offsets, bessel_sum, mask=mask)

def i0(input: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert input.is_cuda, "Input tensor must be on CUDA"

    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_cuda, "Output tensor must be on CUDA"
        assert out.shape == input.shape, "Output tensor shape must match input"

    num_elements = input.numel()
    if num_elements == 0:
        return out  # Handle empty tensor

    input_flat = input.view(-1)
    output_flat = out.view(-1)

    BLOCK_SIZE = 1024  # Optimal block size for element-wise kernels
    K_MAX = 20  # Sufficient for typical use cases

    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    i0_kernel[grid](
        input_flat,
        output_flat,
        num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        K_MAX=K_MAX,
    )

    return out
