import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def _rad2deg_sqrt_kernel(
    input_ptr, out_deg_ptr, out_sqrt_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    deg_val = x * (180.0 / 3.141592653589793)
    sqrt_val = tl.sqrt(x)

    tl.store(out_deg_ptr + offsets, deg_val, mask=mask)
    tl.store(out_sqrt_ptr + offsets, sqrt_val, mask=mask)

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the conversion of angles from radians to degrees and calculates
    the square root for each element in the input tensor.

    Args:
        input (Tensor): The
