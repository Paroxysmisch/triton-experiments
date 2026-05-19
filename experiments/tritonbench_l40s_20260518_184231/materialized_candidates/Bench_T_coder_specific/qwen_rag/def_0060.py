import triton
import triton.language as tl
import torch
import math

# Kernel function to compute the exponential of elements in a tensor
@triton.jit
def exp_kernel(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'a' with boundary mask
    a_value = tl.load(a + offset, mask=mask)
    # Compute the exponential of the loaded elements
    b_value = tl.exp(a_value.to(tl.float32))
    # Store the result in output tensor 'b' with boundary mask
    tl.store(b + offset, b_value, mask=mask)

# Kernel function to compute the square root of elements in a tensor
@triton.jit
def sqrt_kernel(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'a' with boundary mask
    a_value = tl.load(a + offset, mask=mask)
    # Compute the square root of the loaded elements
    b_value = tl.sqrt(a_value.to(tl.float32))
    # Store the result in output tensor 'b' with boundary mask
    tl.store(b + offset, b_value, mask=mask)

# Function to launch the Triton kernel and compute exponential followed by square root
def exp_sqrt(A, out=None):
    # If no output tensor is provided, create one with the same shape as 'A'
    if out is None:
        out = torch.empty_like(A)
    
    # Get the total number of elements in the input tensor
    n_elements = A.numel()
    
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Launch the exponential kernel
    exp_kernel[(grid_size, 1, 1)](A, out, n_elements, block_size)
    
    # Launch the square root kernel on the intermediate result
    sqrt_kernel[(grid_size, 1, 1)](out, out, n_elements, block_size)
    
    return out
