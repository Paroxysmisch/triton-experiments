import triton
import triton.language as tl
import torch
import math

# Kernel function to compute exp followed by sqrt for each element
@triton.jit
def exp_sqrt_kernel(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'a' with boundary mask
    a_value = tl.load(a + offset, mask=mask)
    # Compute the exponential of the loaded elements
    exp_value = tl.exp(a_value.to(tl.float32))
    # Compute the square root of the exponential result
    b_value = tl.sqrt(exp_value)
    # Store the result in output tensor 'b' with boundary mask
    tl.store(b + offset, b_value, mask=mask)

# Function to launch the Triton kernel and compute exp followed by sqrt
def exp_sqrt(input, out=None):
    # Create an output tensor 'out' if not provided
    if out is None:
        out = torch.empty_like(input)
    # Get the total number of elements in the input tensor
    n_elements = input.numel()
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel
    exp_sqrt_kernel[(grid_size,)](input, out, n_elements, block_size)
    return out
