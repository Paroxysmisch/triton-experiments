import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def rad2deg_sqrt_kernel(
    input_ptr, deg_ptr, sqrt_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Define the block index and offset for this program instance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input elements
    input = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Perform the radian to degree conversion and square root calculation
    rad2deg = input * (180.0 / 3.141592653589793)
    sqrt_val = tl.sqrt(input)

    # Store the results
    tl.store(deg_ptr + offsets, rad2deg, mask=offsets < n_elements)
    tl.store(sqrt_ptr + offsets, sqrt_val, mask=offsets < n_elements)

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Ensure the input is a CUDA tensor
    assert input.is_cuda, "Input tensor must be a CUDA tensor"

    # Allocate output tensors
    deg_output = torch.empty_like(input)
    sqrt_output = torch.empty_like(input)

    # Define the number of elements and block size
    n_elements = input.numel()
    BLOCK_SIZE = 1024  # This can be tuned for performance

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    rad2deg_sqrt_kernel[grid](input, deg_output, sqrt_output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return deg_output, sqrt_output

# Example usage:
# input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
# deg_tensor, sqrt_tensor = rad2deg_sqrt(input_tensor)
# print(deg_tensor, sqrt_tensor)
