import triton
import triton.language as tl
import torch

# Triton kernel for seeded dropout
@triton.jit
def _seeded_dropout(
    x_ptr,                # Pointer to the input tensor
    output_ptr,           # Pointer to the output tensor
    n_elements,           # Number of elements to process
    p,                    # Dropout probability
    seed,                 # Seed for random number generation
    BLOCK_SIZE: tl.constexpr  # Size of blocks of data processed by each kernel instance
):
    # Program ID (unique to each kernel instance)
    pid = tl.program_id(0)
    
    # Calculate the range of elements this program instance will handle
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask out-of-bounds elements
    mask = offsets < n_elements
    
    # Load input data (conditionally to avoid out-of-bounds access)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Generate random numbers using a fixed seed for reproducibility
    random_values = tl.rand(seed, offsets)
    
    # Create a dropout mask: keep elements where random values > p
    keep_mask = random_values > p
    
    # Apply the mask and scale the output to maintain mean consistency
    scale = 1.0 / (1.0 - p)
    output = tl.where(keep_mask, x * scale, 0.0)
    
    # Store the result back to the output pointer
    tl.store(output_ptr + offsets, output, mask=mask)

# Python wrapper for the Triton kernel
def seeded_dropout(x, p, seed, block_size=1024):
    """
    Applies seeded dropout to the input tensor `x`.
    
    Args:
        x (torch.Tensor): Input tensor.
        p (float): Dropout probability.
        seed (int): Seed for random number generation.
        block_size (int): Block size for Triton kernel (default: 1024).
    
    Returns:
        torch.Tensor: Tensor with dropout applied.
    """
    # Ensure the input tensor is contiguous
    x = x.contiguous()
    
    # Allocate output tensor of the same shape and type as the input
    output = torch.empty_like(x)
    
    # Get the total number of elements in the input tensor
    n_elements = x.numel()
    
    # Calculate the grid size (number of kernel instances)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the Triton kernel
    _seeded_dropout[grid](
        x_ptr=x, 
        output_ptr=output, 
        n_elements=n_elements, 
        p=p, 
        seed=seed, 
        BLOCK_SIZE=block_size
    )
    
    return output
