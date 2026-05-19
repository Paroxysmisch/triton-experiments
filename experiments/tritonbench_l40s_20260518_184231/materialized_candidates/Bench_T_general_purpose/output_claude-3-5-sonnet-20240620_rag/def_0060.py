import triton
import triton.language as tl
import torch
import math

@triton.jit
def exp_sqrt_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the offset for each block
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for boundary conditions
    mask = offset < n_elements
    
    # Load input elements
    x = tl.load(input_ptr + offset, mask=mask)
    
    # Compute exp(x) followed by sqrt
    result = tl.sqrt(tl.exp(x))
    
    # Store the result
    tl.store(output_ptr + offset, result, mask=mask)

def exp_sqrt(input, out=None) -> torch.Tensor:
    # Input validation
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input)
    elif out.size() != input.size():
        raise ValueError("Output tensor must have the same size as input tensor")
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Calculate block size and grid
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 512))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    exp_sqrt_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
