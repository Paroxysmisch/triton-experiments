import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def erfc_sqrt_kernel(input):
    # Compute the complementary error function (erfc)
    out0 = tl.erfc(input)
    # Compute the square root of each element
    out1 = tl.sqrt(input)
    return out0, out1

def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Call the Triton kernel
    return erfc_sqrt_kernel(input)
