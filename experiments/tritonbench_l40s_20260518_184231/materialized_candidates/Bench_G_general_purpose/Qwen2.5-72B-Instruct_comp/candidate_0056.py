import triton
import triton.language as tl

@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    # Program index for parallel execution
    pid = tl.program_id(0)
    
    # Calculate the offsets for block-wise data processing
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Load data from A and B with masks to prevent out-of-bound access
    a = tl.load(A + offs, mask=offs < size, other=0)
    b = tl.load(B + offs, mask=offs < size, other=0)
    
    # Compute the sum
    c = a + b
    
    # Store the result in C with the same mask condition
    tl.store(C + offs, c, mask=offs < size)

import torch

def custom_add(a, b):
    # Ensure the input tensors are on the same device
    assert a.device == b.device, "Input tensors must be on the same device"
    
    # Initialize the output tensor with the same shape as a
    c = torch.empty_like(a)
    
    # Compute the size of the input tensors
    size = a.numel()
    
    # Define the BLOCK size
    BLOCK = 16
    
    # Calculate the grid size
    grid = (triton.cdiv(size, BLOCK),)
    
    # Launch the kernel
    _add_kernel[grid](a, b, c, size, BLOCK)
    
    return c
