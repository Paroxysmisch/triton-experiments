import triton
import triton.language as tl
import torch

@triton.jit
def asin_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the absolute position of the thread
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute arcsine
    # We use the built-in asin function from Triton language
    result = tl.asin(x)
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

def asin(input, *, out=None):
    """
    Returns a new tensor with the arcsine of the elements of the input tensor.
    
    Args:
        input (Tensor): Input tensor
        out (Tensor, optional): Output tensor to store the result
    
    Returns:
        Tensor: A tensor containing the arcsine of each element in input
    """
    # Input validation
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a tensor")
    
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input)
    elif out.size() != input.size():
        raise ValueError("Output tensor must have the same size as input tensor")
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Define block size (can be tuned for better performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    asin_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
