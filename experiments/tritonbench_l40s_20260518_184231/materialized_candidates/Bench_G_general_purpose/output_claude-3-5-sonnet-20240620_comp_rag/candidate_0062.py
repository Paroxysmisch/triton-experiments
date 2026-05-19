import torch
import triton
import triton.language as tl

# Heuristic to determine optimal block size based on N
@triton.heuristics({'BLOCK': lambda args: min(max(128, triton.next_power_of_2(args['N'])), 1024)})
@triton.jit
def uniform_kernel(
    out_ptr, N, philox_seed, philox_offset,
    from_, to,
    BLOCK: tl.constexpr
):
    # Calculate thread index
    pid = tl.program_id(0)
    block_start = pid * BLOCK * 4  # Each thread handles 4 numbers due to Philox
    offsets = block_start + tl.arange(0, BLOCK * 4)
    
    # Generate random numbers in chunks of 4
    for i in range(0, BLOCK * 4, 4):
        # Calculate offsets for this iteration
        off_0 = offsets + i + 0
        off_1 = offsets + i + 1
        off_2 = offsets + i + 2
        off_3 = offsets + i + 3
        
        # Generate 4 random numbers using Philox
        rand = tl.randint4x(philox_seed, philox_offset + off_0)
        r0, r1, r2, r3 = rand
        
        # Convert to uniform float in [0, 1)
        f0 = tl.uint_to_uniform_float(r0)
        f1 = tl.uint_to_uniform_float(r1)
        f2 = tl.uint_to_uniform_float(r2)
        f3 = tl.uint_to_uniform_float(r3)
        
        # Scale to desired range [from_, to)
        range_size = to - from_
        f0 = from_ + f0 * range_size
        f1 = from_ + f1 * range_size
        f2 = from_ + f2 * range_size
        f3 = from_ + f3 * range_size
        
        # Store results with bounds checking
        mask0 = off_0 < N
        mask1 = off_1 < N
        mask2 = off_2 < N
        mask3 = off_3 < N
        
        tl.store(out_ptr + off_0, f0, mask=mask0, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, f1, mask=mask1, eviction_policy="evict_first")
        tl.store(out_ptr + off_2, f2, mask=mask2, eviction_policy="evict_first")
        tl.store(out_ptr + off_3, f3, mask=mask3, eviction_policy="evict_first")

def uniform_(size, from_=0.0, to=1.0, seed=None, device='cuda'):
    """
    Generate uniform random numbers in the range [from_, to).
    
    Args:
        size: Output tensor size
        from_: Lower bound (inclusive)
        to: Upper bound (exclusive)
        seed: Random seed (optional)
        device: CUDA device to use
    
    Returns:
        Tensor of uniform random numbers
    """
    with torch.cuda.device(device):
        # Initialize output tensor
        out = torch.empty(size, dtype=torch.float32, device=device)
        N = out.numel()
        
        # Initialize Philox state
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, (1,), device=device).item()
        philox_seed, philox_offset = triton.testing.get_philox_seed_offset(seed)
        
        # Calculate grid size (each thread processes 4 numbers)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK'] * 4),)
        
        # Launch kernel
        uniform_kernel[grid](
            out_ptr=out.data_ptr(),
            N=N,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
            from_=from_,
            to=to
        )
        
        return out
