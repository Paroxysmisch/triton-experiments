import triton
import triton.language as tl
import torch

@triton.jit
def sin_kernel(
    in_ptr0,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements to process
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Get the program ID for the current thread block
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets for all elements in this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle edge cases (last block might be partially filled)
    mask = offsets < n_elements
    
    # Load input elements using the calculated offsets
    x = tl.load(in_ptr0 + offsets, mask=mask)
    
    # Compute sine of the input elements
    output = tl.sin(x)
    
    # Store results back to memory
    tl.store(out_ptr + offsets, output, mask=mask)

def sin_triton(x: torch.Tensor, out: torch.Tensor = None):
    """
    Compute sine of input tensor using Triton kernel
    
    Args:
        x: Input tensor
        out: Output tensor (optional)
    
    Returns:
        Tensor containing sine of input elements
    """
    # Handle output tensor
    if out is None:
        out = torch.empty_like(x)
    
    # Get total number of elements
    n_elements = x.numel()
    
    # Define block size (can be tuned for performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    sin_kernel[grid](
        x,  # Input tensor
        out,  # Output tensor
        n_elements,  # Total elements
        BLOCK_SIZE=BLOCK_SIZE,  # Block size
    )
    
    return out
