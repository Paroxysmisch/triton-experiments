import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def erfc_kernel(input, input_size, BLOCK_SIZE: tl.constexpr):
    # Calculate offsets for each block
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offsets < input_size
    # Load elements from input tensor
    x = tl.load(input + offsets, mask=mask)
    # Compute the complementary error function (erfc)
    y = 1 - 2 / tl.constexpr(math.pi) ** 0.5 * tl.libdevice.exp(-x * x)
    # Store the result
    tl.store(input + offsets, y, mask=mask)

@triton.jit
def sqrt_kernel(input, input_size, BLOCK_SIZE: tl.constexpr):
    # Calculate offsets for each block
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offsets < input_size
    # Load elements from input tensor
    x = tl.load(input + offsets, mask=mask)
    # Compute the square root
    y = tl.sqrt(x)
    # Store the result
    tl.store(input + offsets, y, mask=mask)

def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Allocate output tensors
    erfc_out = torch.empty_like(input)
    sqrt_out = torch.empty_like(input)
    # Compute the complementary error function (erfc)
    erfc(input, erfc_out)
    # Compute the square root
    sqrt(input, sqrt_out)
    # Return the results
    return erfc_out, sqrt_out
