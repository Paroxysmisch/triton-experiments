import triton
import triton.language as tl

@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr, y_ptr, z_ptr, output_ptr,
    N, M, BLOCK_SIZE: tl.constexpr
):
    # Define the program ID for grid
    pid = tl.program_id(axis=0)
    
    # Calculate the start position for this block
    block_start = pid * BLOCK_SIZE
    
    # Create a block of indices
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle boundaries
    mask = offsets < N
    
    # Load data from x, y, z
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Perform the fused operation: (x + y) * z and apply activation (ReLU)
    result = (x + y) * z
    result = tl.maximum(result, 0.0)  # ReLU activation
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

import torch

def fused_add_mul_activation_torch(x, y, z):
    # Ensure inputs are on the same device and are contiguous
    assert x.is_cuda and y.is_cuda and z.is_cuda, "Inputs must be CUDA tensors"
    assert x.is_contiguous() and y.is_contiguous() and z.is_contiguous(), "Inputs must be contiguous"
    
    # Get the number of elements
    N = x.numel()
    
    # Define block size
    BLOCK_SIZE = 1024  # You can tune this based on your hardware
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Calculate the number of blocks needed
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch the Triton kernel
    fused_add_mul_activation_kernel[grid](
        x_ptr=x, y_ptr=y, z_ptr=z, output_ptr=output,
        N=N, M=0, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
