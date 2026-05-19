import triton
import triton.language as tl
import torch

@triton.jit
def _seeded_dropout(
    x_ptr,          # pointer to input tensor
    output_ptr,     # pointer to output tensor
    n_elements,     # number of elements in tensor
    p,              # dropout probability
    seed,           # random seed for reproducibility
    BLOCK_SIZE: tl.constexpr,  # size of parallel processing blocks
):
    # Calculate position in the computation
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Generate random numbers using seed
    random = tl.rand(seed, offsets)
    
    # Create dropout mask
    keep_mask = random > p
    
    # Apply dropout and scaling
    output = tl.where(keep_mask, x / (1.0 - p), 0.0)
    
    # Store results
    tl.store(output_ptr + offsets, output, mask=mask)

# Python wrapper function
def seeded_dropout(x: torch.Tensor, p: float, seed: int, BLOCK_SIZE: int = 1024):
    """
    Apply dropout with a fixed seed for reproducibility.
    
    Args:
        x: Input tensor
        p: Dropout probability
        seed: Random seed
        BLOCK_SIZE: Number of elements to process in parallel
    
    Returns:
        Tensor with dropout applied
    """
    # Input validation
    assert 0 <= p < 1, "Dropout probability must be in [0, 1)"
    
    # Prepare output tensor
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Calculate grid size
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    _seeded_dropout[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        p=p,
        seed=seed,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
