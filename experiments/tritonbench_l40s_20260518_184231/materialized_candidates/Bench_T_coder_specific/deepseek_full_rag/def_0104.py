import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def rad2deg_sqrt_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each program instance
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle out-of-bounds accesses
    mask = offset < n_elements
    # Load the input element with the mask
    inp = tl.load(input + offset, mask=mask)
    # Convert radians to degrees and compute the square root
    out0 = inp * (180.0 / 3.141592653589793)
    out1 = tl.sqrt(inp)
    # Store the results
    tl.store(output + offset, out0, mask=mask)
    tl.store(output + offset, out1, mask=mask)

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create an output tensor
    output = torch.empty_like(input)
    # Get the number of elements
    n_elements = input.numel()
    # Define the block size
    BLOCK_SIZE = 1024
    # Calculate the grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    # Launch the Triton kernel
    rad2deg_sqrt_kernel[(grid_size,)](input, output, n_elements, BLOCK_SIZE)
    return output, input
