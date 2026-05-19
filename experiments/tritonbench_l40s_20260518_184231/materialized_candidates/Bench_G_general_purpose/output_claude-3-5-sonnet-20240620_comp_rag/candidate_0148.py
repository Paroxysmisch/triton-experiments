import triton
import triton.language as tl
import torch

@triton.jit
def sin_kernel(
    in_ptr0,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements to process
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per parallel instance
):
    # Get the program instance ID
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this program instance
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load input data with mask to prevent out-of-bounds access
    x = tl.load(in_ptr0 + offsets, mask=mask)
    
    # Compute sine of input values
    output = tl.sin(x)
    
    # Store results back to memory, using mask
    tl.store(out_ptr + offsets, output, mask=mask)

def sin_triton(x: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    """
    Compute element-wise sine using Triton kernel.
    
    Args:
        x: Input tensor
        out: Optional output tensor
    
    Returns:
        Tensor containing element-wise sine of input
    """
    # Handle output tensor
    if out is None:
        out = torch.empty_like(x)
    
    # Get total number of elements
    n_elements = x.numel()
    
    # Compute grid size based on number of elements and block size
    grid = (n_elements + 4 - 1) // 4
    
    # Launch kernel
    sin_kernel[(grid,)](
        x,  # input tensor
        out,  # output tensor
        n_elements,  # total elements
        BLOCK_SIZE=4,  # elements per block
    )
    
    return out
