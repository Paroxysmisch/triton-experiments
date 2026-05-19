import triton
import triton.language as tl
import torch
import math

# Kernel function to compute e^(sqrt(input))
@triton.jit
def exp_func(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor with boundary mask
    input_value = tl.load(input_ptr + offset, mask=mask)
    # Compute the square root of the loaded elements
    sqrt_value = tl.sqrt(input_value.to(tl.float32))
    # Compute the exponential of the square-rooted values
    output_value = tl.exp(sqrt_value)
    # Store the result in output tensor with boundary mask
    tl.store(output_ptr + offset, output_value, mask=mask)

# Wrapper function to launch the Triton kernel
def sqrt_exp(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)  # Create output tensor if not provided
    n_elements = input.numel()  # Get the total number of elements
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))  # Determine block size
    grid_size = triton.cdiv(n_elements, block_size)  # Calculate grid size
    # Launch the Triton kernel
    exp_func[(grid_size, 1, 1)](input, out, n_elements, block_size)
    return out
