import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def _rad2deg_sqrt_kernel(
    input_ptr,
    deg_output_ptr,
    sqrt_output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input = tl.load(input_ptr + offsets, mask=mask)
    # Convert radians to degrees
    rad2deg_factor = 180.0 / tl.math.pi
    deg = input * rad2deg_factor
    # Compute square root
    sqrt_val = tl.sqrt(input)
    # Store results
    tl.store(deg_output_ptr + offsets, deg, mask=mask)
    tl.store(sqrt_output_ptr + offsets, sqrt_val, mask=mask)

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Ensure the input tensor is contiguous
    if not input.is_contiguous():
        input = input.contiguous()
    # Handle 0-dimensional tensor by unsqueezing
    if input.dim() == 0:
        input = input.unsqueeze(0)
    # Allocate output tensors
    deg_output = torch.empty_like(input)
    sqrt_output = torch.empty_like(input)
    n_elements = input.numel()
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = 1024
    _rad2deg_sqrt_kernel[grid](
        input, deg_output, sqrt_output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return (deg_output, sqrt_output)
