import torch
import triton
import triton.language as tl

# Define block size as a power of 2 for optimal performance
BLOCK_SIZE = 1024

@triton.jit
def puzzle1_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size (static)
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    
    # Calculate the block start index
    block_start = pid * BLOCK_SIZE
    
    # Create an offset array for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements (handling edge cases)
    mask = offsets < n_elements
    
    # Load input data using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Perform the addition operation
    output = x + 10
    
    # Store the result using the same mask
    tl.store(out_ptr + offsets, output, mask=mask)

def puzzle1(x: torch.Tensor) -> torch.Tensor:
    # Get input size
    n_elements = x.numel()
    
    # Create output tensor with same shape and dtype as input
    output = torch.empty_like(x)
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    puzzle1_kernel[grid](
        x_ptr=x,
        out_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
