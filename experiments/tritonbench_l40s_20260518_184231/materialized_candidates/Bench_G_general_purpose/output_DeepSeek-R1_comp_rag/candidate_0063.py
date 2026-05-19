import torch
import triton
import triton.language as tl

@triton.jit
def _seeded_dropout(
    x_ptr,          # Pointer to input tensor
    output_ptr,     # Pointer to output tensor
    n_elements,     # Total number of elements in the input tensor
    p,              # Dropout probability (0 <= p < 1)
    seed,           # Seed for random number generation
    BLOCK_SIZE: tl.constexpr,  # Number of elements processed per block
):
    # Determine block offset using program ID
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to handle out-of-bounds elements
    mask = offsets < n_elements
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask)
    # Generate random numbers using the given seed and element offsets
    random = tl.rand(seed, offsets)
    # Determine which elements to keep (random > p)
    x_keep = random > p
    # Apply dropout: zero out elements and scale retained ones
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def seeded_dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    """
    Applies dropout to the input tensor using a specified seed for reproducibility.

    Args:
        x: Input tensor.
        p: Probability of an element being zeroed.
        seed: Seed for random number generation.

    Returns:
        Tensor with dropout applied.
    """
    # Validate dropout probability
    if p < 0 or p >= 1:
        raise ValueError("Dropout probability must be in [0, 1)")
    
    # Ensure input is contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Prepare output tensor
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Handle empty tensor case
    if n_elements == 0:
        return output
    
    # Define block size and compute grid size
    BLOCK_SIZE = 1024  # Optimizable based on hardware
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    
    # Launch the Triton kernel
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=BLOCK_SIZE)
    
    return output
