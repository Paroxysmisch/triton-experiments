import triton
import triton.language as tl

@triton.jit
def uniform_kernel(out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr):
    # Calculate the number of blocks
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK * 4  # Processing in batches of 4
    offsets = block_start + tl.arange(0, BLOCK * 4)

    # Ensure we do not exceed N
    mask = offsets < N

    # Initialize Philox RNG state
    key = philox_seed
    counter = philox_offset + offsets // 4  # Each counter generates 4 random numbers

    # Generate random numbers using Philox
    random_values = tl.philox_4x32(counter, key)

    # Normalize and scale to the desired range
    scale = (to - from_) / (1 << 32)
    offset = from_

    # Compute the uniform random numbers
    uniform_randoms = random_values.to(tl.float32) * scale + offset

    # Write the results to memory
    tl.store(out_ptr + offsets, uniform_randoms, mask=mask)

def generate_uniform_randoms(out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK=256):
    # Calculate the number of blocks needed
    num_blocks = (N + BLOCK * 4 - 1) // (BLOCK * 4)

    # Launch the kernel
    uniform_kernel[(num_blocks,)](out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK)
