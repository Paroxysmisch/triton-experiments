import triton
import triton.language as tl
import torch
from typing import Tuple
import math

# Kernel function to compute the complementary error function (erfc) and square root of elements in a tensor
@triton.jit
def erfc_sqrt_func(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'a' with boundary mask
    a_value = tl.load(a + offset, mask=mask)
    # Compute the square root of the loaded elements
    b_value = tl.sqrt(a_value.to(tl.float32))
    # Compute the complementary error function of the loaded elements
    c_value = tl.erfc(a_value.to(tl.float32))
    # Store the square root result in output tensor 'b' with boundary mask
    tl.store(b + offset, b_value, mask=mask)
    # Store the erfc result in output tensor 'c' with boundary mask
    tl.store(c + offset, c_value, mask=mask)

# Wrapper function to prepare inputs and outputs and launch the Triton kernel
def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create two output tensors 'B' and 'C' with the same shape as the input tensor
    B = torch.empty_like(input)
    C = torch.empty_like(input)
    # Get the total number of elements in the input tensor
    n_elements = input.numel()
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel
    erfc_sqrt_func[(grid_size, 1, 1)](input, B, C, n_elements, block_size)
    # Return a tuple of the two output tensors
    return B, C
