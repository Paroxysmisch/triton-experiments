import triton
import triton.language as tl
import torch
import math
from typing import Tuple

# Kernel function to convert radians to degrees
@triton.jit
def rad2deg_kernel(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'a' with boundary mask
    a_value = tl.load(a + offset, mask=mask)
    # Convert radians to degrees
    b_value = a_value * (180.0 / math.pi)
    # Store the result in output tensor 'b' with boundary mask
    tl.store(b + offset, b_value, mask=mask)

# Kernel function to compute the square root of elements in a tensor
@triton.jit
def sqrt_kernel(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'a' with boundary mask
    a_value = tl.load(a + offset, mask=mask)
    # Compute the square root of the loaded elements
    b_value = tl.sqrt(a_value.to(tl.float32))
    # Store the result in output tensor 'b' with boundary mask
    tl.store(b + offset, b_value, mask=mask)

# Wrapper function to convert radians to degrees and compute square root
def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create an output tensor for degrees with the same shape as 'input'
    degrees = torch.empty_like(input)
    # Create an output tensor for square roots with the same shape as 'input'
    square_roots = torch.empty_like(input)
    
    # Get the total number of elements in the input tensor
    n_elements = input.numel()
    
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Launch the rad2deg kernel
    rad2deg_kernel[(grid_size, 1, 1)](input, degrees, n_elements, block_size)
    
    # Launch the sqrt kernel
    sqrt_kernel[(grid_size, 1, 1)](degrees, square_roots, n_elements, block_size)
    
    return degrees, square_roots

# Example usage
if __name__ == "__main__":
    input_tensor = torch.tensor([0.0, math.pi/6, math.pi/4, math.pi/2], dtype=torch.float32)
    degrees, square_roots = rad2deg_sqrt(input_tensor)
    print("Degrees:", degrees)
    print("Square Roots:", square_roots)
