import torch
import triton
import triton.language as tl
from typing import Tuple

# Triton kernel for computing the square root
@triton.jit
def sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Obtain the program ID
    pid = tl.program_id(axis=0)
    
    # Define offsets for the current block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load input values, masking out-of-bounds
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute the square root
    sqrt_vals = tl.sqrt(input_vals)
    
    # Store the result
    tl.store(output_ptr + offsets, sqrt_vals, mask=mask)

# Wrapper function
def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Check input tensor is on the correct device
    assert input.is_cuda, "Input tensor must be on CUDA device"

    # Compute erfc using PyTorch
    erfc_result = torch.erfc(input)
    
    # Prepare output tensor for square root
    sqrt_result = torch.empty_like(input)
    
    # Launch the Triton kernel
    n_elements = input.numel()
    BLOCK_SIZE = 1024  # You can adjust this based on your GPU capabilities
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    sqrt_kernel[grid](input, sqrt_result, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return erfc_result, sqrt_result

# Example usage
input_tensor = torch.tensor([0.1, 0.5, 1.0, 2.0], device='cuda')
erfc_res, sqrt_res = erfc_sqrt(input_tensor)
print("ERFC:", erfc_res)
print("SQRT:", sqrt_res)
