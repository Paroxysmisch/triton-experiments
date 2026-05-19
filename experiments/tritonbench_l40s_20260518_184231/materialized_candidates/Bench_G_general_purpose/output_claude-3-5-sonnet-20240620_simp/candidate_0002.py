import torch
import triton
import triton.language as tl

@triton.jit
def kldivergence_kernel(
    x_ptr,  # pointer to first input vector (x)
    y_ptr,  # pointer to second input vector (y)
    output_ptr,  # pointer to output vector
    n_elements,  # number of elements in the vectors
    BLOCK_SIZE: tl.constexpr,  # size of the block to process
):
    # Calculate the program ID and the block start index
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create a mask for valid elements (handles edge cases)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Add small epsilon to prevent division by zero and log(0)
    eps = 1e-10
    x = x + eps
    y = y + eps
    
    # Compute KL divergence: x * log(x/y)
    kl = x * tl.log(x / y)
    
    # Store the result
    tl.store(output_ptr + offsets, kl, mask=mask)

def kldivergence(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between two probability distributions.
    
    Args:
        x: First probability distribution (torch.Tensor)
        y: Second probability distribution (torch.Tensor)
    
    Returns:
        torch.Tensor: KL divergence between x and y
    """
    # Input validation
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.is_cuda and y.is_cuda, "Input tensors must be on GPU"
    assert x.dtype == y.dtype, "Input tensors must have the same dtype"
    
    # Get input size
    n_elements = x.numel()
    
    # Initialize output tensor
    output = torch.empty_like(x)
    
    # Define block size (can be tuned for performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    kldivergence_kernel[grid](
        x.data_ptr(),
        y.data_ptr(),
        output.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return sum of KL divergence
    return output.sum()
