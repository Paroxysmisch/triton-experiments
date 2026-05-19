import torch
import triton
import triton.language as tl
import math

@triton.jit
def erf_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute block start/end indices
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Constants for Abramowitz and Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    # Take the absolute value
    sign = tl.where(x < 0, -1, 1)
    x = tl.abs(x)
    
    # Formula: erf(x) = 1 - (1 / (1 + px)^16) * exp(-x^2)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * tl.exp(-x * x)
    
    # Apply sign
    result = sign * y
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def erf(input, *, out=None):
    """
    Computes the error function of the input tensor.
    
    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: A tensor containing the error function of each element in input
    """
    # Input validation
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a tensor")
    
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input)
    elif out.size() != input.size():
        raise RuntimeError("Output tensor must be same size as input tensor")
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Configure block size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    erf_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
