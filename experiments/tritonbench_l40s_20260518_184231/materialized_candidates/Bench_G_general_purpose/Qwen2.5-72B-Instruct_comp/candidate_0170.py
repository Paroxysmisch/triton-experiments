import triton
import triton.language as tl

@triton.jit
def cos_func(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the starting point for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Create a range of offsets for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to ensure we only process valid elements
    mask = offsets < n_elements

    # Load the elements from the input tensor `a`
    a_value = tl.load(a_ptr + offsets, mask=mask)

    # Compute the cosine of each element
    b_value = tl.cos(a_value)

    # Store the results in the output tensor `b`
    tl.store(b_ptr + offsets, b_value, mask=mask)

import torch
import triton
import triton.language as tl

def cos(a: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    a = a.cuda()
    
    # Create the output tensor with the same shape and type as the input tensor
    b = torch.empty_like(a)
    
    # Calculate the block size as the nearest power of 2 greater than the square root of the number of elements
    n_elements = a.numel()
    block_size = 1
    while block_size * block_size < n_elements:
        block_size *= 2
    
    # Determine the grid size to cover all elements
    grid_size = (n_elements + block_size - 1) // block_size
    
    # Execute the Triton kernel
    cos_func[grid_size, block_size](a, b, n_elements, BLOCK_SIZE=block_size)
    
    return b
