import triton
import triton.language as tl
import torch
import math

# Kernel function to compute the sqrt followed by tanh of elements in a tensor
@triton.jit
def sqrt_tanh_kernel(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'a' with boundary mask
    a_value = tl.load(a + offset, mask=mask)
    # Compute the square root of the loaded elements
    sqrt_value = tl.sqrt(a_value.to(tl.float32))
    # Apply tanh to the square-rooted values
    tanh_value = tl.tanh(sqrt_value)
    # Store the result in output tensor 'b' with boundary mask
    tl.store(b + offset, tanh_value, mask=mask)

# Wrapper function to launch the Triton kernel and compute sqrt followed by tanh
def sqrt_tanh(input, out=None):
    # If no output tensor is provided, create one with the same shape as 'input'
    if out is None:
        out = torch.empty_like(input)
    # Get the total number of elements in the input tensor
    n_elements = input.numel()
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel
    sqrt_tanh_kernel[(grid_size,)](input, out, n_elements, block_size)
    return out
