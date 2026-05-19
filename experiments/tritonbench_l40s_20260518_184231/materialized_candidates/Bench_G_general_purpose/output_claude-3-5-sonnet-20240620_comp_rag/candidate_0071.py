import triton
import triton.language as tl
import torch

# Kernel for dropout using precomputed mask
@triton.jit
def _triton_dropout(
    x_ptr,          # pointer to input tensor
    x_keep_ptr,     # pointer to precomputed mask (0s and 1s)
    output_ptr,     # pointer to output tensor
    n_elements,     # number of elements in tensor
    p,              # dropout probability
    BLOCK_SIZE: tl.constexpr,  # size of parallel blocks
):
    # Calculate offsets for this program instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input and mask data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)

    # Apply dropout using precomputed mask
    output = tl.where(x_keep, x / (1 - p), 0.0)
    
    # Store results
    tl.store(output_ptr + offsets, output, mask=mask)

# Kernel for seeded dropout
@triton.jit
def _seeded_triton_dropout(
    x_ptr,          # pointer to input tensor
    output_ptr,     # pointer to output tensor
    n_elements,     # number of elements in tensor
    p,              # dropout probability
    seed,           # random seed for reproducibility
    BLOCK_SIZE: tl.constexpr,  # size of parallel blocks
):
    # Calculate offsets for this program instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask)

    # Generate random mask using seed
    random = tl.rand(seed, offsets)
    x_keep = random > p

    # Apply dropout with scaling
    output = tl.where(x_keep, x / (1 - p), 0.0)
    
    # Store results
    tl.store(output_ptr + offsets, output, mask=mask)

# Python wrapper for precomputed mask dropout
def triton_dropout(x: torch.Tensor, mask: torch.Tensor, p: float) -> torch.Tensor:
    """
    Apply dropout using a precomputed mask.
    
    Args:
        x: Input tensor
        mask: Binary mask tensor (1s for keep, 0s for drop)
        p: Dropout probability
    Returns:
        Tensor with dropout applied
    """
    # Ensure input is contiguous
    x = x.contiguous()
    mask = mask.contiguous()
    
    # Initialize output tensor
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Configure block size and grid
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    _triton_dropout[grid](
        x.data_ptr(),
        mask.data_ptr(),
        output.data_ptr(),
        n_elements,
        p,
        BLOCK_SIZE,
    )
    
    return output

# Python wrapper for seeded dropout
def seeded_triton_dropout(x: torch.Tensor, p: float, seed: int = None) -> torch.Tensor:
    """
    Apply dropout with a random seed.
    
    Args:
        x: Input tensor
        p: Dropout probability
        seed: Random seed for reproducibility
    Returns:
        Tensor with dropout applied
    """
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Initialize output tensor
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Use current time as seed if none provided
    if seed is None:
        seed = int(torch.randint(0, 2**31-1, (1,)).item())
    
    # Configure block size and grid
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    _seeded_triton_dropout[grid](
        x.data_ptr(),
        output.data_ptr(),
        n_elements,
        p,
        seed,
        BLOCK_SIZE,
    )
    
    return output
