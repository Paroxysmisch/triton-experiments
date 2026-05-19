import triton
import triton.language as tl
import torch
import math
from typing import Tuple

# Kernel function to convert radians to degrees and compute the square root
@triton.jit
def rad2deg_sqrt_kernel(input, degrees_output, sqrt_output, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    
    # Load elements from input tensor 'input' with boundary mask
    input_value = tl.load(input + offset, mask=mask)
    
    # Compute degrees and square root
    degrees_value = input_value * (180.0 / math.pi)
    sqrt_value = tl.sqrt(input_value.to(tl.float32))
    
    # Store the results in output tensors with boundary mask
    tl.store(degrees_output + offset, degrees_value, mask=mask)
    tl.store(sqrt_output + offset, sqrt_value, mask=mask)

# Wrapper function to launch the Triton kernel
def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create output tensors for degrees and square roots
    degrees_output = torch.empty_like(input)
    sqrt_output = torch.empty_like(input)
    
    # Get the total number of elements in the input tensor
    n_elements = input.numel()
    
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Launch the Triton kernel
    rad2deg_sqrt_kernel[(grid_size, 1, 1)](input, degrees_output, sqrt_output, n_elements, block_size)
    
    return degrees_output, sqrt_output
