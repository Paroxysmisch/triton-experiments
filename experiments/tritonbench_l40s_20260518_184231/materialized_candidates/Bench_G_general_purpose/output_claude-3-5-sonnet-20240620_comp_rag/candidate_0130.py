import triton
import triton.language as tl
import torch

@triton.jit
def kernel_function(
    x_ptr,  # Pointer to input
    output_ptr,  # Pointer to output  
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Static block size
):
    # Calculate starting point for this program instance
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offset tensor for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input data using mask
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute sine
    output = tl.math.sin(x)
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def call_kernel(x: torch.Tensor) -> torch.Tensor:
    # Input validation and setup
    assert x.is_contiguous()
    n_elements = x.numel()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    kernel_function[grid](
        x_ptr=x.data_ptr(),
        output_ptr=output.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
