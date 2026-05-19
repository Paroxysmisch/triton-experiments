import torch
import triton
import triton.language as tl

@triton.jit
def puzzle1_kernel(
    x_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    value,  # Constant value to add
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Calculate starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input data using mask
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Add constant value
    output = x + value
    
    # Store result using mask
    tl.store(output_ptr + offsets, output, mask=mask)

def puzzle1(x: torch.Tensor, value: float = 10.0) -> torch.Tensor:
    # Input validation
    assert x.is_cuda, "Input tensor must be on GPU"
    
    # Create output tensor with same properties as input
    output = torch.empty_like(x)
    
    # Get total number of elements
    n_elements = x.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    puzzle1_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        value=value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
