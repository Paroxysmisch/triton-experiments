import torch
import triton
import triton.language as tl

@triton.jit
def cos_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    # Calculate the starting offset for this program instance
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask for valid elements
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    # Compute cosine
    output = tl.cos(x)
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def cos(input, *, out=None):
    """
    Returns a new tensor with the cosine of the elements of the input tensor.
    
    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: A tensor containing the cosine of each element in input
    """
    # Input validation
    assert input.is_cuda, "Input tensor must be on GPU"
    
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_cuda, "Output tensor must be on GPU"
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
    
    # Calculate total number of elements
    n_elements = input.numel()
    
    # Calculate block size (power of 2)
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    cos_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
