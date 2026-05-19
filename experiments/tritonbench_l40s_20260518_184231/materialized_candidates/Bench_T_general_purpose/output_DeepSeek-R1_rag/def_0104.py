import triton
import triton.language as tl
import torch
import math
from typing import Tuple

@triton.jit
def rad2deg_sqrt_kernel(input_ptr, deg_ptr, sqrt_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_val = tl.load(input_ptr + offsets, mask=mask)
    # Convert radians to degrees
    deg_val = input_val * (180.0 / tl.math.pi)
    # Compute square root
    sqrt_val = tl.sqrt(input_val)
    # Store results
    tl.store(deg_ptr + offsets, deg_val, mask=mask)
    tl.store(sqrt_ptr + offsets, sqrt_val, mask=mask)

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Ensure input is contiguous and flattened for 1D processing
    input_contiguous = input.contiguous().view(-1)
    n_elements = input_contiguous.numel()
    # Create output tensors
    deg_output = torch.empty_like(input_contiguous)
    sqrt_output = torch.empty_like(input_contiguous)
    # Compute optimal block size and grid size
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the kernel
    rad2deg_sqrt_kernel[(grid_size, 1, 1)](input_contiguous, deg_output, sqrt_output, n_elements, block_size)
    # Reshape outputs to match the original input shape
    deg_output = deg_output.view_as(input)
    sqrt_output = sqrt_output.view_as(input)
    return (deg_output, sqrt_output)
