import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    output_ptr,
    input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the pid and the number of elements to process
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Apply ReLU: max(0, x)
    output = tl.maximum(0, x)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def relu(input, inplace=False):
    """
    Applies ReLU function element-wise: ReLU(x) = max(0, x)
    
    Args:
        input: Input tensor
        inplace: If True, modifies the input tensor directly. Default: False
    
    Returns:
        Tensor with ReLU applied element-wise
    """
    # Handle inplace operation
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Define block size (a power of 2)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    relu_kernel[grid](
        output,
        input,
        n_elements,
        BLOCK_SIZE,
    )
    
    return output
