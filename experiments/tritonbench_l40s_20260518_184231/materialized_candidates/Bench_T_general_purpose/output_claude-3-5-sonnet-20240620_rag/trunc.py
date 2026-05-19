import torch
import triton
import triton.language as tl

@triton.jit
def trunc_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Calculate the start offset for this program instance
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for bounds checking
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute trunc
    # For floating point numbers, we use floor for positive numbers and ceil for negative numbers
    result = tl.where(x >= 0, 
                     tl.floor(x),
                     tl.ceil(x))
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

def trunc(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """
    Returns a new tensor with the truncated integer values of the elements of input.
    
    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: A tensor containing the truncated values
    """
    # If input is integer type, return a copy as per array-api convention
    if torch.is_integer_dtype(input.dtype):
        return input.clone() if out is None else out.copy_(input)
    
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    trunc_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
