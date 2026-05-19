import triton
import triton.language as tl

@triton.jit
def gammaln_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Calculate the absolute position of the block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Take absolute value of input
    x = tl.abs(x)
    
    # Lanczos approximation for ln(Gamma(x))
    # Constants for Lanczos approximation
    g = 7
    p = tl.zeros_like(x) + 0.99999999999980993
    
    # Coefficients for Lanczos approximation
    coeff = tl.zeros_like(x) + 676.5203681218851
    p = p + coeff / (x + 0)
    coeff = tl.zeros_like(x) - 1259.1392167224028
    p = p + coeff / (x + 1)
    coeff = tl.zeros_like(x) + 771.32342877765313
    p = p + coeff / (x + 2)
    coeff = tl.zeros_like(x) - 176.61502916214059
    p = p + coeff / (x + 3)
    coeff = tl.zeros_like(x) + 12.507343278686905
    p = p + coeff / (x + 4)
    coeff = tl.zeros_like(x) - 0.13857109526572012
    p = p + coeff / (x + 5)
    coeff = tl.zeros_like(x) + 9.9843695780195716e-6
    p = p + coeff / (x + 6)
    
    # Calculate final result
    t = x + g - 0.5
    result = tl.log(2.506628275 * p) + (x - 0.5) * tl.log(t) - t
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

import torch
import triton
import math

def gammaln(input, *, out=None):
    """
    Computes the natural logarithm of the absolute value of the gamma function on the input tensor.
    
    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: A tensor containing the natural logarithm of the absolute gamma function
    """
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input)
    elif out.shape != input.shape:
        raise RuntimeError("Output tensor must have the same shape as input tensor")
    
    # Get tensor properties
    n_elements = input.numel()
    
    # Define block size (can be tuned for performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    gammaln_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
