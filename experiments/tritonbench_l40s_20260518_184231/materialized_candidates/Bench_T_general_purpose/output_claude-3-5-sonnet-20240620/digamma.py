import torch
import triton
import triton.language as tl

@triton.jit
def digamma_kernel(
    x_ptr,  # pointer to input tensor
    out_ptr,  # pointer to output tensor
    n_elements,  # number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # number of elements to process per block
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    # Compute the block start index
    block_start = pid * BLOCK_SIZE
    # Compute offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle the case where block is not full
    mask = offsets < n_elements
    
    # Load data for this block
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Constants for the digamma approximation
    c = 12.0
    euler_mascheroni = 0.57721566490153286060
    
    # Initialize the result
    result = tl.zeros_like(x)
    
    # Handle special case x == 0
    zero_mask = x == 0
    result = tl.where(zero_mask, float('-inf'), result)
    
    # Handle x > 0
    pos_mask = (~zero_mask) & mask
    x_pos = tl.where(pos_mask, x, 1.0)  # Use 1.0 as placeholder for masked values
    
    # For x < 0.5, use reflection formula
    reflection_mask = (x_pos < 0.5) & pos_mask
    x_reflected = tl.where(reflection_mask, 1.0 - x_pos, x_pos)
    
    # Main approximation
    # Use Stirling's approximation for large values
    s = tl.zeros_like(x_pos)
    x_temp = x_reflected
    
    # Add terms until x_temp > c
    while tl.sum(x_temp <= c):
        s = tl.where(x_temp <= c, s - 1.0/x_temp, s)
        x_temp = tl.where(x_temp <= c, x_temp + 1.0, x_temp)
    
    # Compute approximation for large values
    x_temp = x_temp - 0.5
    s = s + tl.log(x_temp) + 1.0/(2.0*x_temp)
    
    # Apply reflection formula where needed
    s = tl.where(reflection_mask, 
                 -s - tl.pi/tl.tan(tl.pi * x_pos),
                 s)
    
    # Store final result
    result = tl.where(pos_mask, s, result)
    tl.store(out_ptr + offsets, result, mask=mask)

def digamma(input, *, out=None):
    """
    Computes the logarithmic derivative of the gamma function.
    
    Args:
        input (Tensor): the tensor to compute the digamma function on
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: A tensor containing the digamma function computed on input
    """
    if out is None:
        out = torch.empty_like(input)
    
    # Handle empty tensor
    if input.numel() == 0:
        return out
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(input.numel(), meta['BLOCK_SIZE']),)
    digamma_kernel[grid](input, out, input.numel(), BLOCK_SIZE=1024)
    
    return out
