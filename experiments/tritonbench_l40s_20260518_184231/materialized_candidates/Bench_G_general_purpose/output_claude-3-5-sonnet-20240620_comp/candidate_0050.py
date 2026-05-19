import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Calculate starting position for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input data using the mask
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Apply ReLU: max(0, x)
    output = tl.where(x > 0, x, 0.0)
    
    # Store results back to memory
    tl.store(output_ptr + offsets, output, mask=mask)

def relu(x):
    """
    Wrapper function for the ReLU Triton kernel
    Args:
        x: Input tensor
    Returns:
        Output tensor after applying ReLU
    """
    # Get input properties
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Define block size (can be tuned for performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    relu_kernel[grid](
        x,
        output,
        n_elements,
        BLOCK_SIZE,
    )
    
    return output
