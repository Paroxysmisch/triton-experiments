import triton
import triton.language as tl
import torch

@triton.jit
def kldivergence_kernel(
    x_ptr,                                # Pointer to input x
    y_ptr,                                # Pointer to input y
    output_ptr,                           # Pointer to output
    n_elements,                           # Total number of elements
    BLOCK_SIZE: tl.constexpr             # Size of the block to process
):
    # Get program ID and compute block start index
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load x and y values
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute KL divergence: x * log(x/y)
    # Add small epsilon to prevent division by zero and log(0)
    epsilon = 1e-10
    kl_div = x * tl.log((x + epsilon) / (y + epsilon))
    
    # Store the result
    tl.store(output_ptr + offsets, kl_div, mask=mask)

def kldivergence(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between two probability distributions x and y.
    
    Args:
        x: Input tensor (probability distribution)
        y: Input tensor (probability distribution)
    
    Returns:
        Tensor containing element-wise KL divergence
    """
    # Input validation
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.is_cuda and y.is_cuda, "Input tensors must be on GPU"
    assert x.is_contiguous() and y.is_contiguous(), "Input tensors must be contiguous"
    
    # Get input size
    n_elements = x.numel()
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    kldivergence_kernel[grid](
        x.data_ptr(),
        y.data_ptr(),
        output.data_ptr(),
        n_elements,
        BLOCK_SIZE
    )
    
    return output
