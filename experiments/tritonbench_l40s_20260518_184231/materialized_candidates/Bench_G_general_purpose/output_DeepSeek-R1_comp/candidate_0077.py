import triton
import triton.language as tl

@triton.jit
def _seeded_dropout_kernel(
    x_ptr,          # Pointer to input tensor
    output_ptr,      # Pointer to output tensor
    n_elements,      # Total number of elements in the tensor
    p,               # Dropout probability (float)
    seed,            # Seed for reproducibility
    BLOCK_SIZE: tl.constexpr,  # Number of elements processed per block
):
    # Determine block index and element offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle out-of-bounds elements
    mask = offsets < n_elements
    
    # Load data from input tensor
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Generate deterministic random numbers using seed and element offsets
    rand = tl.rand(seed, offsets)
    keep_mask = rand > p  # True = keep element
    
    # Apply dropout: zero out elements and scale survivors by 1/(1-p)
    output = tl.where(keep_mask, x / (1 - p), 0.0)
    
    # Store results to output tensor
    tl.store(output_ptr + offsets, output, mask=mask)

import torch

def seeded_dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    """
    Applies dropout with seed-based determinism to input tensor `x`.
    
    Args:
        x: Input tensor (GPU-based).
        p: Probability of dropping an element (0 <= p < 1).
        seed: Seed for random number generation.
    
    Returns:
        torch.Tensor: Tensor with dropout applied.
    """
    # Ensure input is on GPU and contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.is_cuda, "Input tensor must reside on GPU"
    assert 0 <= p < 1, "Dropout probability must be in [0, 1)"
    
    # Prepare output tensor
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Handle empty tensor edge case
    if n_elements == 0:
        return output
    
    # Kernel configuration
    BLOCK_SIZE = 1024  # Elements per block
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE  # Compute grid size
    
    # Launch kernel
    _seeded_dropout_kernel[grid](
        x, output, n_elements, p, seed, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Create a sample tensor on GPU
x = torch.randn(4096, device='cuda')

# Apply dropout with p=0.5 and seed=42
output = seeded_dropout(x, p=0.5, seed=42)

# Verify approx 50% of elements are zeroed
print(f"Non-zero elements: {(output != 0).float().mean()}")
