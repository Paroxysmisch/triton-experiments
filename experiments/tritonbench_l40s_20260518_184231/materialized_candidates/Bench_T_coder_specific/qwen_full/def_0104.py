import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def _rad2deg_sqrt_triton(input):
    out1 = input * (180.0 / 3.141592653589793)
    out2 = tl.sqrt(input)
    return out1, out2

def rad2deg_sqrt_triton(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    if input.is_floating_point():
        out1, out2 = _rad2deg_sqrt_triton(input)
        return out1, out2
    else:
        raise ValueError("Input must be a floating-point tensor")
