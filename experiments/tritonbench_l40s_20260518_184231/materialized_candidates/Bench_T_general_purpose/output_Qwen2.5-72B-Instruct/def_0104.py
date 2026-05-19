import triton
import triton.language as tl

@triton.jit
def rad2deg_sqrt_kernel(
    input_ptr,  # Pointer to the input tensor
    output_deg_ptr,  # Pointer to the output tensor for degrees
    output_sqrt_ptr,  # Pointer to the output tensor for square roots
    n_elements,  # Number of elements in the input tensor
    BLOCK_SIZE: tl.constexpr  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    
    # Convert radians to degrees
    deg_vals = input_vals * (180.0 / 3.141592653589793)
    # Compute square roots
    sqrt_vals = tl.sqrt(input_vals)
    
    tl.store(output_deg_ptr + offsets, deg_vals, mask=mask)
    tl.store(output_sqrt_ptr + offsets, sqrt_vals, mask=mask)

import torch
from typing import Tuple
import triton
import triton.language as tl

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Ensure the input tensor is on the same device as the Triton kernel
    device = input.device
    input = input.contiguous()
    
    # Allocate output tensors
    output_deg = torch.empty_like(input)
    output_sqrt = torch.empty_like(input)
    
    # Launch the Triton kernel
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    rad2deg_sqrt_kernel[grid](
        input, output_deg, output_sqrt, n_elements, BLOCK_SIZE=1024
    )
    
    return output_deg, output_sqrt

import torch

# Test input tensor
input_tensor = torch.tensor([0.0, 3.141592653589793, 1.5707963267948966], device='cuda')

# Call the wrapper function
output_deg, output_sqrt = rad2deg_sqrt(input_tensor)

# Print the results
print("Degrees:", output_deg)
print("Square Roots:", output_sqrt)
