import triton
import triton.language as tl
import torch

@triton.jit
def sqrt_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Calculate the absolute position of the block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate the offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements
    mask = offsets < n_elements
    
    # Load input values using the mask
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute square root
    output = tl.sqrt(x)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def sqrt(input, *, out=None):
    """
    Returns a new tensor with the square-root of the elements of the input tensor.
    
    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: A tensor containing the square root of each element in input
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
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Define block size (can be tuned for better performance)
    BLOCK_SIZE = 1024
    
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
