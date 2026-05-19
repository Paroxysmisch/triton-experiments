import triton
import triton.language as tl
import torch
import math
from typing import Tuple

# Kernel function to compute the complementary error function (erfc) of elements in a tensor
@triton.jit
def erfc_kernel(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a_value = tl.load(a + offset, mask=mask)
    # Use the approximation for erfc
    b_value = 1 - tl.erf(a_value.to(tl.float32))
    tl.store(b + offset, b_value, mask=mask)

# Kernel function to compute the square root of elements in a tensor
@triton.jit
def sqrt_kernel(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a_value = tl.load(a + offset, mask=mask)
    b_value = tl.sqrt(a_value.to(tl.float32))
    tl.store(b + offset, b_value, mask=mask)

# Wrapper function to compute both erfc and sqrt
def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create output tensors
    erfc_output = torch.empty_like(input)
    sqrt_output = torch.empty_like(input)
    
    # Get the total number of elements in the input tensor
    n_elements = input.numel()
    
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Launch the Triton kernels
    erfc_kernel[(grid_size,)](input, erfc_output, n_elements, block_size)
    sqrt_kernel[(grid_size,)](input, sqrt_output, n_elements, block_size)
    
    return erfc_output, sqrt_output

# Example usage:
input_tensor = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float32)
erfc_result, sqrt_result = erfc_sqrt(input_tensor)
print("Erfc:", erfc_result)
print("Sqrt:", sqrt_result)
