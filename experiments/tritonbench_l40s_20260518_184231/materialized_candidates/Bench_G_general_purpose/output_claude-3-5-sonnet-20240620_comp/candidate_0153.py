import triton
import triton.language as tl
import torch

@triton.jit
def sin_kernel(
    in_ptr0,  # Pointer to input array
    out_ptr,  # Pointer to output array
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this program instance
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load input data with mask
    x = tl.load(in_ptr0 + offsets, mask=mask)
    
    # Compute sine
    output = tl.sin(x)
    
    # Store result with mask
    tl.store(out_ptr + offsets, output, mask=mask)

def sin_triton(x: torch.Tensor) -> torch.Tensor:
    # Input validation
    assert x.is_contiguous(), "Input tensor must be contiguous"
    
    # Get input shape and create output tensor
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Define block size
    BLOCK_SIZE = 4
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    sin_kernel[grid](
        x.data_ptr(),
        output.data_ptr(),
        n_elements,
        BLOCK_SIZE,
    )
    
    return output
