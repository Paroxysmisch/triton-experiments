import torch
import triton
import triton.language as tl
import math

@triton.jit
def i0_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    # Calculate offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create mask for bounds checking
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Initialize sum for Bessel function
    result = tl.zeros_like(x)
    x_squared_div_4 = (x * x) / 4.0
    
    # We'll use a fixed number of terms for approximation
    # Usually 10-15 terms give good accuracy for moderate inputs
    term = tl.ones_like(x)
    factorial_k = 1.0
    
    # Sum first 12 terms of the series
    for k in range(12):
        if k > 0:
            factorial_k *= k
            term = term * x_squared_div_4 / (factorial_k * factorial_k)
        result += term
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def i0(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the zeroth order modified Bessel function of the first kind for each element.
    
    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: A tensor containing the zeroth order modified Bessel function of the first kind
    """
    # Input validation
    assert input.is_cuda, "Input tensor must be on GPU"
    
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_cuda, "Output tensor must be on GPU"
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Calculate block size (power of 2)
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    i0_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
