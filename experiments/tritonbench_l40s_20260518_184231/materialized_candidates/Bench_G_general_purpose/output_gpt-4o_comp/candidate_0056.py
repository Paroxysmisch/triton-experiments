import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    # Program ID determines which block of data this instance will handle
    prog_id = tl.program_id(0)
    
    # Calculate the offset for the current block
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    
    # Load data from A and B with boundary checks using masks
    mask = offs < size
    a_vals = tl.load(A + offs, mask=mask, other=0.0)
    b_vals = tl.load(B + offs, mask=mask, other=0.0)
    
    # Perform element-wise addition
    c_vals = a_vals + b_vals
    
    # Store the result in C
    tl.store(C + offs, c_vals, mask=mask)

def custom_add(a, b):
    # Ensure the inputs are PyTorch tensors and have the same shape
    assert a.shape == b.shape, "Input tensors must have the same shape"
    
    # Determine the size of the input tensors
    size = a.numel()
    
    # Initialize the output tensor with the same shape and device as the inputs
    c = torch.empty_like(a)
    
    # Define the BLOCK size for processing
    BLOCK = 16
    
    # Calculate the number of blocks needed to cover the entire tensor
    grid = (triton.cdiv(size, BLOCK),)
    
    # Launch the Triton kernel
    _add_kernel[grid](a, b, c, size, BLOCK)
    
    # Return the result tensor
    return c

# Example usage
a = torch.randn(128, device='cuda')
b = torch.randn(128, device='cuda')
c = custom_add(a, b)
print(c)
