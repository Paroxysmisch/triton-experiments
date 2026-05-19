import triton
import triton.language as tl
import torch
import math

@triton.jit
def cos_func_kernel(
    b_ptr,  # Pointer to output tensor
    a_ptr,  # Pointer to input tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Static block size parameter
):
    # Calculate the starting offset for this program instance
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for bounds checking
    mask = offset < n_elements
    
    # Load input values using the mask
    a_value = tl.load(a_ptr + offset, mask=mask)
    
    # Compute cosine
    b_value = tl.cos(a_value)
    
    # Store results using the mask
    tl.store(b_ptr + offset, b_value, mask=mask)

def cos(x: torch.Tensor) -> torch.Tensor:
    """
    Compute element-wise cosine of input tensor using Triton.
    
    Args:
        x: Input tensor
    Returns:
        Output tensor containing cos(x)
    """
    # Make sure input is contiguous and on GPU
    x = x.contiguous()
    assert x.is_cuda, "Input tensor must be on GPU"
    
    # Prepare output tensor
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Calculate optimal block size (nearest power of 2 to sqrt(n))
    block_size = 2 ** int(math.log2(math.sqrt(n_elements)) + 1)
    block_size = min(block_size, 1024)  # Ensure block size doesn't exceed GPU limits
    
    # Calculate grid size to cover all elements
    grid_size = (n_elements + block_size - 1) // block_size
    
    # Launch kernel
    cos_func_kernel[(grid_size,)](
        output,
        x,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    
    return output
