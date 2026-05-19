import triton
import triton.language as tl
import torch
import math

@triton.jit
def uniform_kernel(
    out_ptr,  # Pointer to output tensor
    N,        # Total number of elements
    philox_seed,    # Random seed
    philox_offset,  # Random offset
    from_,    # Lower bound
    to,       # Upper bound
    BLOCK: tl.constexpr,  # Block size
):
    # Calculate pid and number of elements to process per thread
    pid = tl.program_id(0)
    block_start = pid * BLOCK * 4  # *4 because we generate 4 numbers per iteration
    
    # Convert seed and offset to int32
    seed = tl.zeros([2], dtype=tl.int32)
    seed[0] = philox_seed
    seed[1] = philox_offset + block_start
    
    # Loop over block with stride 4 (Philox generates 4 numbers at once)
    for i in range(0, BLOCK, 1):
        offset = block_start + i * 4
        
        # Generate 4 random numbers using Philox
        rand = tl.philox(seed[0], seed[1] + i)
        r0, r1, r2, r3 = rand[0], rand[1], rand[2], rand[3]
        
        # Convert to float in [0, 1) range
        f0 = tl.uint_to_uniform_float(r0)
        f1 = tl.uint_to_uniform_float(r1)
        f2 = tl.uint_to_uniform_float(r2)
        f3 = tl.uint_to_uniform_float(r3)
        
        # Scale to desired range [from_, to)
        range_size = to - from_
        f0 = f0 * range_size + from_
        f1 = f1 * range_size + from_
        f2 = f2 * range_size + from_
        f3 = f3 * range_size + from_
        
        # Calculate offsets for the 4 values
        off_0 = offset + 0
        off_1 = offset + 1
        off_2 = offset + 2
        off_3 = offset + 3
        
        # Store values if within bounds
        if off_0 < N:
            tl.store(out_ptr + off_0, f0, eviction_policy="evict_first")
        if off_1 < N:
            tl.store(out_ptr + off_1, f1, eviction_policy="evict_first")
        if off_2 < N:
            tl.store(out_ptr + off_2, f2, eviction_policy="evict_first")
        if off_3 < N:
            tl.store(out_ptr + off_3, f3, eviction_policy="evict_first")

def uniform_(size, from_=0., to=1., device=None):
    """
    Generate uniform random numbers in the range [from_, to)
    
    Args:
        size: Output tensor size or shape
        from_: Lower bound (default: 0.0)
        to: Upper bound (default: 1.0)
        device: torch device (default: None, uses current device)
    
    Returns:
        torch.Tensor: Tensor filled with random numbers
    """
    # Handle input size
    if isinstance(size, int):
        N = size
    else:
        N = math.prod(size)
    
    # Get current device if none specified
    if device is None:
        device = torch.cuda.current_device()
    
    # Create output tensor
    out = torch.empty(N, device=device, dtype=torch.float32)
    
    # Initialize Philox state
    seed = torch.randint(0, 2**32, (1,), device=device).item()
    offset = torch.randint(0, 2**32, (1,), device=device).item()
    
    # Calculate grid size and block size
    BLOCK = min(triton.next_power_of_2(N // 4), 2048)
    grid = (triton.cdiv(N, BLOCK * 4),)
    
    # Launch kernel
    uniform_kernel[grid](
        out_ptr=out,
        N=N,
        philox_seed=seed,
        philox_offset=offset,
        from_=from_,
        to=to,
        BLOCK=BLOCK,
        num_warps=4,
    )
    
    # Reshape if necessary
    if isinstance(size, tuple):
        out = out.reshape(size)
    
    return out
