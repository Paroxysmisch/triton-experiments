import triton
import triton.language as tl

@triton.jit
def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the program's ID for the block dimension
    pid = tl.program_id(0)
    
    # Compute the base index for each block
    block_start = pid * BLOCK_SIZE
    
    # Compute the range of indices for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to prevent out-of-bound memory access
    mask = offsets < n_elements
    
    # Load the input tensors
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    
    # Perform the element-wise addition
    output = x + y
    
    # Store the result back to the output tensor
    tl.store(out_ptr + offsets, output, mask=mask)

import torch

def add_wrapper(x, y):
    # Initialize the output tensor with the same shape and dtype as x
    out = torch.zeros_like(x)
    
    # Determine the total number of elements
    n_elements = x.numel()
    
    # Define the block size
    BLOCK_SIZE = 1024
    
    # Compute the number of blocks needed
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE)
    
    return out
