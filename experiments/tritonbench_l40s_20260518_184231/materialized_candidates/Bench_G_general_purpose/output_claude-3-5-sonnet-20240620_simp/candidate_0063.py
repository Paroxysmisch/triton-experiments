import triton
import triton.language as tl

@triton.jit
def uniform_kernel(
    out_ptr,  # Pointer to output array
    N,        # Number of elements
    philox_seed,    # Random seed
    philox_offset,  # Offset for random generation
    from_,    # Lower bound (inclusive)
    to,       # Upper bound (exclusive)
    BLOCK: tl.constexpr,  # Block size for parallel execution
):
    # Calculate the program ID and the total number of blocks
    pid = tl.program_id(0)
    grid = (N + BLOCK - 1) // BLOCK
    
    # Calculate the start offset for this program instance
    offset = pid * BLOCK
    
    # Initialize Philox key
    key = tl.zeros([2], dtype=tl.uint32)
    key[0] = philox_seed
    counter = tl.zeros([4], dtype=tl.uint32)
    
    # Process elements in blocks
    for idx in range(offset, offset + BLOCK, 4):
        # Skip if we're beyond the array bounds
        if idx >= N:
            break
            
        # Set counter for this batch
        counter[0] = idx + philox_offset
        
        # Generate random numbers using Philox
        rand = tl.philox4x32(counter, key)
        
        # Convert to float32 and scale to [0, 1)
        rand_f32 = tl.uint32_to_uniform_float32(rand)
        
        # Scale to desired range [from_, to)
        range_size = to - from_
        rand_scaled = from_ + rand_f32 * range_size
        
        # Store results with bounds checking
        mask = tl.arange(0, 4) + idx < N
        tl.store(out_ptr + idx, rand_scaled, mask=mask)

# Python wrapper function
def uniform_random(shape, from_=0.0, to=1.0, seed=None, device='cuda'):
    """
    Generate uniform random numbers in the range [from_, to).
    
    Args:
        shape: Output shape
        from_: Lower bound (inclusive)
        to: Upper bound (exclusive)
        seed: Random seed (optional)
        device: Device to run on
        
    Returns:
        Tensor of random numbers
    """
    import torch
    
    # Calculate total number of elements
    N = torch.prod(torch.tensor(shape)).item()
    
    # Initialize output tensor
    output = torch.empty(shape, device=device, dtype=torch.float32)
    
    # Set seed and offset
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    offset = 0
    
    # Calculate grid and block sizes
    BLOCK = 1024
    grid = (N + BLOCK - 1) // BLOCK
    
    # Launch kernel
    uniform_kernel[(grid,)](
        output.data_ptr(),
        N,
        seed,
        offset,
        from_,
        to,
        BLOCK
    )
    
    return output
