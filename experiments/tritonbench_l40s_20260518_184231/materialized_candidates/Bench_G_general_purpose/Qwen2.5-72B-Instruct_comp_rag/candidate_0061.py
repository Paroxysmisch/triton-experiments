import torch
import triton
import triton.language as tl

# Define the block size
BLOCK = 1024

# Triton kernel for generating uniform random numbers
@triton.jit
def uniform_kernel(out_ptr, N, philox_seed, philox_offset, from_, to_, BLOCK: tl.constexpr):
    # Calculate the starting index for this block
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    
    # Generate four random numbers using Philox
    rand = tl.rand(philox_seed, philox_offset + offsets)
    r0, r1, r2, r3 = rand[0], rand[1], rand[2], rand[3]
    
    # Convert uints to floats in the [0, 1) range
    r0 = tl.uint_to_uniform_float(r0)
    r1 = tl.uint_to_uniform_float(r1)
    r2 = tl.uint_to_uniform_float(r2)
    r3 = tl.uint_to_uniform_float(r3)
    
    # Scale to the specified range [from_, to_)
    r0 = from_ + (to_ - from_) * r0
    r1 = from_ + (to_ - from_) * r1
    r2 = from_ + (to_ - from_) * r2
    r3 = from_ + (to_ - from_) * r3
    
    # Calculate the offsets for storing the random numbers
    off_0 = block_start + 0 * BLOCK
    off_1 = block_start + 1 * BLOCK
    off_2 = block_start + 2 * BLOCK
    off_3 = block_start + 3 * BLOCK
    
    # Store the random numbers in the output array
    tl.store(out_ptr + off_0, r0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, r1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, r2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, r3, mask=off_3 < N, eviction_policy="evict_first")

def uniform_(out, from_, to_, philox_seed, philox_offset):
    # Ensure the output tensor is on the current CUDA device
    device = out.device
    with torch.cuda.device(device):
        # Calculate the total number of elements
        N = out.numel()
        
        # Initialize the Philox generator state
        philox_seed, philox_offset = triton.rand.philox_cuda_seed_offset(philox_seed, philox_offset)
        
        # Calculate the grid size
        grid = (triton.cdiv(N, BLOCK),)
        
        # Invoke the kernel
        uniform_kernel[grid](out, N, philox_seed, philox_offset, from_, to_, BLOCK)
