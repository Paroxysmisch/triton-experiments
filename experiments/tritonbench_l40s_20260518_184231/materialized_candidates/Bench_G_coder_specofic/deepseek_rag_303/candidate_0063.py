import torch
import triton
import triton.language as tl

BLOCK = 1024

# Triton kernel for uniform PRNG
@triton.jit
def uniform_kernel(
    out_ptr,  # Pointer to the output array
    N,  # Total number of elements to generate
    philox_seed,  # Seed for Philox random number generator
    philox_offset,  # Offset for Philox random number generator
    from_, to,  # Range of numbers to generate
    **META  # Meta-parameters for kernel tuning
):
    # Initialize offsets
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    off_0 = i4
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    # Generate random uints
    r0, r1, r2, r3 = tl.philox(philox_seed, philox_offset, off_0, off_1, off_2, off_3)

    # Convert to uniform floats in [0, 1) and scale to [from_, to)
    float0 = tl.uint_to_uniform_float(r0) * (to - from_) + from_
    float1 = tl.uint_to_uniform_float(r1) * (to - from_) + from_
    float2 = tl.uint_to_uniform_float(r2) * (to - from_) + from_
    float3 = tl.uint_to_uniform_float(r3) * (to - from_) + from_

    # Store results in out_ptr
    eviction_policy = "evict_first"
    num_stores = 0
    if i4[0] < N:
        tl.store(out_ptr + off_0, float0, eviction_policy=eviction_policy)
        num_stores += 1
    if i4[0] + BLOCK < N:
        tl.store(out_ptr + off_1, float1, eviction_policy=eviction_policy)
        num_stores += 1
    if i4[0] + 2 * BLOCK < N:
        tl.store(out_ptr + off_2, float2, eviction_policy=eviction_policy)
        num_stores += 1
    if i4[0] + 3 * BLOCK < N:
        tl.store(out_ptr + off_3, float3, eviction_policy=eviction_policy)
        num_stores += 1

# Function to call the Triton kernel
def uniform_(self, *, from_=0.0, to=1.0, generator=None):
    # Initialize Philox generator
    philox_seed, philox_offset = philox_cuda_seed_offset(CUDA_SEED_COUNTER)
    increment_cuda_seed_counter()

    N = self.numel()
    if N <= 0:
        return
    
    # Heuristics to tune BLOCK and num_warps
    if N < 4 * 1024:
        BLOCK = 2 * 1024
    elif N < 8 * 1024:
        BLOCK = 4 * 1024
    elif N < 16 * 1024:
        BLOCK = 8 * 1024
    else:
        BLOCK = 16 * 1024

    num_warps = 4
    if N < 4 * 1024:
        num_warps = 2
    elif N < 8 * 1024:
        num_warps = 4
    elif N < 16 * 1024:
        num_warps = 8

    # Call Triton kernel
    with torch.cuda.device(self.device):
        uniform_kernel[(ceildiv(N, BLOCK),)](self, N, philox_seed, philox_offset, from_, to)
