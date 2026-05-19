import triton
import triton.language as tl

@triton.jit
def log_kernel(
    x_ptr,  # pointer to input tensor
    y_ptr,  # pointer to output tensor
    n_elements,  # number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # block size for parallelization
):
    # Compute the pid (program ID)
    pid = tl.program_id(axis=0)
    
    # Compute the block start and offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements
    mask = offsets < n_elements
    
    # Load input values using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute natural logarithm
    y = tl.log(x)
    
    # Store the result
    tl.store(y_ptr + offsets, y, mask=mask)

import triton
import torch

def log(input, *, out=None):
    """
    Returns a new tensor with the natural logarithm of the elements of the input tensor.
    
    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: A tensor containing the natural logarithm of each element in input
    """
    # Input validation
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a tensor")
    
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input)
    elif not isinstance(out, torch.Tensor):
        raise TypeError("out must be a tensor")
    elif out.size() != input.size():
        raise ValueError("out must have the same size as input")
    
    # Get input properties
    n_elements = input.numel()
    
    # Define block size (can be tuned for better performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    log_kernel[grid](
        input.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE,
    )
    
    return out
