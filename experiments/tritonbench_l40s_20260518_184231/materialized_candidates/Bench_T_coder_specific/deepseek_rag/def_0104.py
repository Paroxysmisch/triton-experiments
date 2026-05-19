import triton.language as tl
import torch
import math

# Kernel function to compute the conversion from radians to degrees and square root of elements in a tensor
@triton.jit
def rad2deg_sqrt_kernel(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions
    mask = offset < n_elements
    # Load elements from input tensor 'a' with boundary mask
    a_value = tl.load(a + offset, mask=mask)
    # Compute the conversion from radians to degrees
    b_value_deg = a_value * (180.0 / math.pi)
    # Compute the square root of the loaded elements
    b_value_sqrt = tl.sqrt(a_value.to(tl.float32))
    # Store the result in output tensor 'b' with boundary mask
    tl.store(b + offset, b_value_deg, mask=mask)
    tl.store(b + offset, b_value_sqrt, mask=mask)

# Function to launch the Triton kernel and compute rad2deg and sqrt
def rad2deg_sqrt(A):
    # Create an output tensor 'B' with the same shape as 'A'
    B = torch.empty_like(A)
    # Get the total number of elements in the input tensor
    n_elements = A.numel()
    # Determine the block size for the kernel
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Calculate the grid size for the kernel launch
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel
    rad2deg_sqrt_kernel[(grid_size, 1, 1)](A, B, n_elements, block_size)
    return B
