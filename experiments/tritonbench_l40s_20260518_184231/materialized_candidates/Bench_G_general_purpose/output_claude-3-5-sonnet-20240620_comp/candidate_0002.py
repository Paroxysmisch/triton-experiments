import torch
import triton
import triton.language as tl

@triton.jit
def kldivergence_kernel(
    x_ptr,  # pointer to first input vector (P)
    y_ptr,  # pointer to second input vector (Q) 
    output_ptr,  # pointer to output vector
    n_elements,  # number of elements in the vectors
    BLOCK_SIZE: tl.constexpr,  # size of the block
):
    # Compute unique program ID
    pid = tl.program_id(axis=0)
    
    # Calculate starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offset tensor for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load data for this block
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute KL divergence: x * log(x/y)
    # Add small epsilon to prevent division by zero and log(0)
    eps = 1e-10
    x = x + eps
    y = y + eps
    output = x * tl.log(x / y)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def kldivergence(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between two probability distributions.
    
    Args:
        x: First input tensor (P distribution)
        y: Second input tensor (Q distribution)
    
    Returns:
        torch.Tensor: KL divergence result
    """
    # Input validation
    assert x.is_cuda and y.is_cuda, "Inputs must be GPU tensors"
    assert x.shape == y.shape, "Input shapes must match"
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    kldivergence_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
