import triton
import triton.language as tl
import torch
import math
from typing import Tuple

@triton.jit
def _rad2deg_sqrt_kernel(
    inp_ptr, deg_ptr, sqrt_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    val = tl.load(inp_ptr + offset, mask=mask)
    deg_val = val * (180.0 / 3.141592653589793)
    sqrt_val = tl.sqrt(val)
    tl.store(deg_ptr + offset, deg_val, mask=mask)
    tl.store(sqrt_ptr + offset, sqrt_val, mask=mask)

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    deg_out = torch.empty_like(input)
    sqrt_out = torch.empty_like(input)
    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    _rad2deg_sqrt_kernel[(grid_size,)](
        input, deg_out, sqrt_out,
        n_elements,
        block_size
    )
    return deg_out, sqrt_out
