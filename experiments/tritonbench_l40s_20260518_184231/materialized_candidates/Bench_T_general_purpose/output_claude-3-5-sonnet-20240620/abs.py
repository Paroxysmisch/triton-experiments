import triton
import triton.language as tl
import torch

@triton.jit
def abs_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Calculate the absolute position of the block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Load the input data
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute absolute value
    output = tl.abs(x)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def abs(input, *, out=None):
    """
    Computes the absolute value of each element in the input tensor.
    
    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: A tensor containing the absolute value of each element in input
    """
    # Input validation
    assert isinstance(input, torch.Tensor), "Input must be a tensor"
    
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input"
    
    # Get tensor properties
    n_elements = input.numel()
    
    # Define block size (can be tuned for better performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    abs_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
