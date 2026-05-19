import triton
import triton.language as tl
import torch
import math

@triton.jit
def relu_sqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate offset for this program instance
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for bounds checking
    mask = offset < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offset, mask=mask)
    
    # Apply ReLU (max(0, x))
    x = tl.maximum(x, 0.0)
    
    # Compute square root
    result = tl.sqrt(x.to(tl.float32))
    
    # Store result
    tl.store(output_ptr + offset, result, mask=mask)

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
    elif out.shape != input.shape:
        raise ValueError("Output tensor must have the same shape as input tensor")
    
    # If not inplace and no out provided, we need to copy input
    if not inplace and out is input:
        input = input.clone()
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Calculate block size and grid
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    relu_sqrt_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
