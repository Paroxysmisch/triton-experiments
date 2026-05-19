import torch
import triton
import triton.language as tl
import math

@triton.jit
def airy_ai_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Calculate offsets for this program instance
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute Airy Ai function using series approximation
    # For x < 0, we use a different approximation than for x >= 0
    
    # Initialize result
    result = tl.zeros_like(x)
    
    # Constants for the approximation
    c1 = 0.355028053887817
    c2 = 0.258819403792807
    
    # Compute for x >= 0
    pos_mask = x >= 0
    where_pos = tl.where(pos_mask)
    if tl.sum(pos_mask) > 0:
        z = tl.pow(x[where_pos], 1.5) * 2.0/3.0
        result[where_pos] = c1 * tl.exp(-z) / tl.pow(x[where_pos] + 1e-8, 0.25)
    
    # Compute for x < 0
    neg_mask = ~pos_mask
    where_neg = tl.where(neg_mask)
    if tl.sum(neg_mask) > 0:
        z = tl.abs(x[where_neg])
        result[where_neg] = c2 * tl.cos(2.0/3.0 * tl.pow(z, 1.5) - math.pi/4)
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

def airy_ai(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """
    Computes the Airy function Ai for each element of the input tensor.
    
    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: A tensor containing the Airy Ai function values
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
    airy_ai_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
