import triton
import triton.language as tl
import torch
import math

# Kernel function to compute tanh(sqrt(input))
@triton.jit
def sqrt_tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'input' with boundary mask
    input_value = tl.load(input_ptr + offset, mask=mask)
    # Compute the square root of the loaded elements
    sqrt_value = tl.sqrt(input_value.to(tl.float32))
    # Apply the hyperbolic tangent function
    output_value = tl.tanh(sqrt_value)
    # Store the result in output tensor 'output' with boundary mask
    tl.store(output_ptr + offset, output_value, mask=mask)

# Wrapper function to launch the Triton kernel
def sqrt_tanh(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    
    # Get the total number of elements in the input tensor
    n_elements = input.numel()
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel
    sqrt_tanh_kernel[(grid_size, 1, 1)](input, out, n_elements, block_size)
    return out
