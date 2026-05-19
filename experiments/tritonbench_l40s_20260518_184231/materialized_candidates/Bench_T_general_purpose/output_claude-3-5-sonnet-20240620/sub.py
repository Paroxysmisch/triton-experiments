import triton
import triton.language as tl

@triton.jit
def sub_kernel(
    input_ptr,    # Pointer to input tensor
    other_ptr,    # Pointer to other tensor
    output_ptr,   # Pointer to output tensor
    alpha,        # Alpha scaling factor
    n_elements,   # Number of elements to process
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Calculate the absolute position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for bounds checking
    mask = offsets < n_elements
    
    # Load input and other tensors
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)
    
    # Perform the computation: out = input - alpha * other
    output = x - alpha * y
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

import triton
import torch

def sub(input, other, *, alpha=1, out=None):
    """
    Subtracts other, scaled by alpha, from input.
    
    Args:
        input (Tensor): the input tensor
        other (Tensor or Number): the tensor or number to subtract from input
        alpha (Number, optional): the multiplier for other. Default: 1
        out (Tensor, optional): the output tensor. Default: None
    
    Returns:
        Tensor: The result of the subtraction
    """
    # Handle scalar other
    if not isinstance(other, torch.Tensor):
        other = torch.full_like(input, other)
    
    # Broadcast tensors if needed
    input, other = torch.broadcast_tensors(input, other)
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    elif out.shape != input.shape:
        raise RuntimeError("out tensor must have same shape as input")
    
    # Ensure all tensors are contiguous
    input = input.contiguous()
    other = other.contiguous()
    out = out.contiguous()
    
    # Get tensor properties
    n_elements = input.numel()
    
    # Define block size (can be tuned for better performance)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    sub_kernel[grid](
        input_ptr=input,
        other_ptr=other,
        output_ptr=out,
        alpha=alpha,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
