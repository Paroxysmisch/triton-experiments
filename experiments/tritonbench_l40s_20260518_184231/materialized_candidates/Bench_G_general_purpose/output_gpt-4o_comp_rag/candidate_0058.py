import torch
import triton
import triton.language as tl

BLOCK = 1024

@triton.jit
def uniform_kernel(out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK * 4

    for i in range(0, BLOCK):
        idx = block_start + i * 4

        if idx >= N:
            break

        # Generate 4 random uints using Philox
        r0, r1, r2, r3 = tl.philox(philox_seed, philox_offset + idx)

        # Convert uints to floats in [0, 1)
        u0 = tl.uint_to_uniform_float(r0)
        u1 = tl.uint_to_uniform_float(r1)
        u2 = tl.uint_to_uniform_float(r2)
        u3 = tl.uint_to_uniform_float(r3)

        # Scale to [from_, to)
        f0 = from_ + (to - from_) * u0
        f1 = from_ + (to - from_) * u1
        f2 = from_ + (to - from_) * u2
        f3 = from_ + (to - from_) * u3

        # Calculate offsets
        off_0 = idx
        off_1 = idx + 1
        off_2 = idx + 2
        off_3 = idx + 3

        # Store results conditionally
        if off_0 < N:
            tl.store(out_ptr + off_0, f0, eviction_policy="evict_first")
        if off_1 < N:
            tl.store(out_ptr + off_1, f1, eviction_policy="evict_first")
        if off_2 < N:
            tl.store(out_ptr + off_2, f2, eviction_policy="evict_first")
        if off_3 < N:
            tl.store(out_ptr + off_3, f3, eviction_policy="evict_first")

def uniform_(out_ptr, N, philox_seed, philox_offset, from_, to, device='cuda'):
    grid = (triton.cdiv(N, BLOCK * 4),)
    with torch.cuda.device(device):
        uniform_kernel[grid](out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK)

# Example usage
N = 10240
out = torch.empty(N, dtype=torch.float32, device='cuda')
philox_seed = 12345
philox_offset = 0
from_ = 0.0
to = 1.0

uniform_(out.data_ptr(), N, philox_seed, philox_offset, from_, to)

print(out)
