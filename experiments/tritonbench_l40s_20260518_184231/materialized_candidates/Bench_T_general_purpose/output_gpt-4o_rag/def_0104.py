import triton
import triton.language as tl
import torch
import math
from typing import Tuple

# Kernel function to convert radians to degrees and compute the square root
@triton.jit
def rad2deg_sqrt_kernel(a, deg_out, sqrt_out, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'a' with boundary mask
    a_value = tl.load(a + offset, mask=mask)
    # Convert radians to degrees
    deg_value = a_value * (180.0 / math.pi)
    # Compute the square root of the loaded elements
    sqrt_value = tl.sqrt(a_value.to(tl.float32))
    # Store the result in output tensors with boundary mask
    tl.store(deg_out + offset, deg_value, mask=mask)
    tl.store(sqrt_out + offset, sqrt_value, mask=mask)

# Wrapper function to launch the Triton kernel
def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create output tensors for degrees and square root
    deg_out = torch.empty_like(input)
    sqrt_out = torch.empty_like(input)
    # Get the total number of elements in the input tensor
    n_elements = input.numel()
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel
    rad2deg_sqrt_kernel[(grid_size,)](input, deg_out, sqrt_out, n_elements, block_size)
    return deg_out, sqrt_out
