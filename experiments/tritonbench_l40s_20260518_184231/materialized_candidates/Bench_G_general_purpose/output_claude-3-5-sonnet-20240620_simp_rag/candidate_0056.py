import triton
import triton.language as tl
import torch

@triton.jit
def uniform_kernel(
    out_ptr, # Pointer to output tensor
    N, # Number of elements
    philox_seed, # Random seed
    philox_offset, # Offset for random number generation
    from_, # Lower bound (inclusive)
    to, # Upper bound (exclusive)
    BLOCK: tl.constexpr, # Block size
):
    # Calculate the starting position for this program instance
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    
    # Generate offsets for this block
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < N
    
    # Initialize philox state for random number generation
    state = tl.philox_init(philox_seed, offsets + philox_offset)
    
    # Generate 4 random numbers at a time (Philox4x32 algorithm)
    rand4 = tl.philox_next(state)
    
    # Convert uint32 to float in [0, 1) and scale to desired range
    range_size = to - from_
    rand = from_ + range_size * tl.uint32_to_uniform_float(rand4)
    
    # Store the results
    tl.store(out_ptr + offsets, rand, mask=mask)

def uniform_random(
    size, 
    from_=0.0, 
    to=1.0, 
    seed=None, 
    device='cuda'
):
    """
    Generate uniform random numbers in the range [from_, to).
    
    Args:
        size: Tuple or int specifying output tensor shape
        from_: Lower bound (inclusive)
        to: Upper bound (exclusive)
        seed: Random seed (optional)
        device: Device to place the output tensor on
        
    Returns:
        torch.Tensor: Tensor of uniform random numbers
    """
    # Handle size specification
    if isinstance(size, int):
        size = (size,)
    
    # Create output tensor
    out = torch.empty(size, dtype=torch.float32, device=device)
    N = out.numel()
    
    # Set seed if not provided
    if seed is None:
        seed = torch.randint(0, 2**63 - 1, (1,)).item()
    
    # Calculate grid size
    grid = (triton.cdiv(N, BLOCK),)
    
    # Launch kernel
    uniform_kernel[grid](
        out_ptr=out, 
        N=N,
        philox_seed=seed,
        philox_offset=0,
        from_=from_,
        to=to,
        BLOCK=BLOCK,
    )
    
    return out
