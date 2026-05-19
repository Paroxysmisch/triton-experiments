import triton
import triton.language as tl
import torch
import math

# Kernel function to compute the square root of elements in a tensor
@triton.jit
def sqrt_kernel(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the global index within the block
    pid = tl.program_id(0)
    # Calculate the starting index for this block
    start_idx = pid * BLOCK_SIZE
    # Calculate the indices for this block
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    # Mask to handle boundary conditions
    mask = offsets < n_elements
    # Load elements from input tensor 'a' with boundary mask
    a_values = tl.load(a + offsets, mask=mask)
    # Compute the square root of the loaded elements
    b_values = tl.sqrt(a_values)
    # Store the result in output tensor 'b' with boundary mask
    tl.store(b + offsets, b_values, mask=mask)

# Wrapper function to launch the Triton kernel and compute square root
def sqrt(input, out=None):
    # If no output tensor is provided, create one
    if out is None:
        out = torch.empty_like(input)
    
    # Get the total number of elements in the input tensor
    n_elements = input.numel()
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Launch the Triton kernel
    sqrt_kernel[(grid_size, 1, 1)](input, out, n_elements, block_size)
    
    return out
