import triton
import triton.language as tl

@triton.jit
def uniform_kernel(out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK
    rng = tl.rand.philox(philox_seed, philox_offset + block_start)

    for i in range(block_start, min(block_start + BLOCK, N)):
        # Generate 4 random numbers at once
        rand_nums = rng.next4()
        # Convert to float32 and scale to [0, 1)
        rand_nums = tl.cast(rand_nums, tl.float32) / 4294967296.0
        # Scale to [from_, to)
        rand_nums = from_ + rand_nums * (to - from_)
        # Write to memory
        out_ptr[i] = rand_nums[0]
        if i + 1 < N:
            out_ptr[i + 1] = rand_nums[1]
        if i + 2 < N:
            out_ptr[i + 2] = rand_nums[2]
        if i + 3 < N:
            out_ptr[i + 3] = rand_nums[3]

import triton
import triton.language as tl
import numpy as np
import torch

def generate_uniform_random_numbers(N, from_, to, device='cuda'):
    # Allocate output tensor
    out = torch.empty(N, dtype=torch.float32, device=device)
    
    # Define block size
    BLOCK = 1024
    
    # Compute number of blocks
    num_blocks = (N + BLOCK - 1) // BLOCK
    
    # Define Philox seed and offset
    philox_seed = np.array([0, 0], dtype=np.uint64)
    philox_offset = np.array([0, 0], dtype=np.uint64)
    
    # Launch the kernel
    uniform_kernel[(num_blocks,)](out, N, philox_seed, philox_offset, from_, to, BLOCK)
    
    return out

# Example usage
N = 10000
from_ = 0.0
to = 1.0
random_numbers = generate_uniform_random_numbers(N, from_, to)
print(random_numbers)
