import triton
import triton.language as tl
import torch
import math

@triton.jit
def sqrt_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Size of the block for parallel execution
):
    # Calculate offset for each thread
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for bounds checking
    mask = offsets < n_elements
    
    # Load input values using the mask
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute square root
    # Convert to float32 for computation
    result = tl.sqrt(x.to(tl.float32))
    
    # Store results back to output tensor
    tl.store(output_ptr + offsets, result, mask=mask)

def sqrt(input, *, out=None):
    """
    Computes the square root of elements in the input tensor.
    
    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: A tensor containing the square root of each element in input
    """
    # Input validation
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a tensor")
    
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input)
    elif out.size() != input.size():
        raise RuntimeError("Output tensor must have the same size as input tensor")
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Calculate block size (power of 2)
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 512))
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    sqrt_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
