import torch
import triton
import triton.language as tl
import math

@triton.jit
def gammaln_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Compute the offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.abs(tl.load(input_ptr + offsets, mask=mask))
    
    # Constants for Lanczos approximation
    g = 7
    p = tl.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ])
    
    # Compute log gamma
    result = tl.zeros_like(x)
    
    # x <= 0 cases are handled by taking absolute value earlier
    t = x + g + 0.5
    
    # Calculate sum for Lanczos approximation
    sum_term = p[0]
    for i in range(1, 9):
        sum_term += p[i] / (x + tl.full_like(x, i))
    
    # Compute result using Lanczos formula
    result = (x + 0.5) * tl.log(t) - t + 0.5 * tl.log(2 * 3.141592653589793) + tl.log(sum_term)
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

def gammaln(input, *, out=None):
    """
    Computes the natural logarithm of the absolute value of the gamma function on the input tensor.
    
    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: A tensor containing the natural logarithm of the absolute value of gamma function
    """
    if out is None:
        out = torch.empty_like(input)
    elif out.shape != input.shape:
        raise RuntimeError("out tensor must have same shape as input tensor")
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Calculate block size (power of 2)
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 512))
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    gammaln_kernel[grid](
        out,
        input,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
