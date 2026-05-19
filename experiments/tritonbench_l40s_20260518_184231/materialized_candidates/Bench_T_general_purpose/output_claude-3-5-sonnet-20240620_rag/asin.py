import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def asin_kernel(
    x_ptr,    # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the absolute position in the tensor
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle bounds checking
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute arcsine using libdevice
    result = libdevice.asin(x)
    
    # Store the result
    tl.store(out_ptr + offsets, result, mask=mask)

def asin(input, *, out=None):
    """
    Returns a new tensor with the arcsine of the elements of input.
    
    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor. If provided, the result will be written to this tensor.
    
    Returns:
        Tensor: A tensor containing the arcsine of each element in input.
    """
    # Input validation
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a tensor")
    
    # Ensure input is on CUDA
    if not input.is_cuda:
        raise ValueError("input tensor must be on CUDA device")
    
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input)
    elif not out.is_cuda:
        raise ValueError("output tensor must be on CUDA device")
    elif out.size() != input.size():
        raise ValueError("output tensor must be same size as input tensor")
    
    # Get tensor elements
    n_elements = input.numel()
    
    # Define block size and grid
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    asin_kernel[grid](
        input,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
