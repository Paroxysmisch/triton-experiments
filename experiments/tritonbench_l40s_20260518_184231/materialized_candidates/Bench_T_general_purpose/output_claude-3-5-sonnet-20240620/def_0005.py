import triton
import triton.language as tl

@triton.jit
def relu_sqrt_kernel(
    x_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements in tensor
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
):
    # Calculate the absolute position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Apply ReLU (max(0, x))
    x = tl.maximum(0.0, x)
    
    # Apply square root
    output = tl.sqrt(x)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

import triton
import torch

def relu_sqrt(input, inplace=False, out=None) -> torch.Tensor:
    # Input validation
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a tensor")
    
    # Handle output tensor
    if out is None:
        if inplace:
            out = input
        else:
            out = torch.empty_like(input)
    
    # Get input shape and size
    n_elements = input.numel()
    
    # Define block size (can be tuned for better performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    relu_sqrt_kernel[grid](
        input.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE,
    )
    
    return out
