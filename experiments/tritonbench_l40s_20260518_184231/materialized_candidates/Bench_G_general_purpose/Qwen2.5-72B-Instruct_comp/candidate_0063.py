import triton
import triton.language as tl

@triton.jit
def uniform_kernel(
    out_ptr,  # Pointer to the output array
    N,        # Total number of random numbers to generate
    philox_seed,  # Seed for the Philox random number generator
    philox_offset,  # Offset for the Philox random number generator
    from_,    # Lower bound of the range
    to,       # Upper bound of the range
    BLOCK: tl.constexpr  # Block size
):
    # Compute the number of threads in the block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)

    # Initialize the Philox state
    philox_state = tl.philox_state(philox_seed, philox_offset)

    # Loop over the block
    for i in range(0, N, BLOCK * 4):
        # Generate 4 random numbers per thread
        r0, r1, r2, r3 = tl.philox4(philox_state)

        # Convert to uniform float in [0, 1)
        u0 = tl.uint_to_uniform_float(r0)
        u1 = tl.uint_to_uniform_float(r1)
        u2 = tl.uint_to_uniform_float(r2)
        u3 = tl.uint_to_uniform_float(r3)

        # Scale to the specified range [from_, to)
        u0 = from_ + u0 * (to - from_)
        u1 = from_ + u1 * (to - from_)
        u2 = from_ + u2 * (to - from_)
        u3 = from_ + u3 * (to - from_)

        # Calculate offsets for each random number
        off_0 = offsets + i
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK

        # Conditionally store the random numbers
        tl.store(out_ptr + off_0, u0, eviction_policy="evict_first")
        tl.store(out_ptr + off_1, u1, eviction_policy="evict_first")
        tl.store(out_ptr + off_2, u2, eviction_policy="evict_first")
        tl.store(out_ptr + off_3, u3, eviction_policy="evict_first")

        # Update the Philox state
        philox_state = tl.philox_advance(philox_state, 4 * BLOCK)

    # Ensure all writes are completed
    tl.device_barrier()

import torch
import triton
import triton.runtime as triton_runtime

def uniform_(out, from_, to, philox_seed, philox_offset):
    # Ensure the output tensor is on the current CUDA device
    device = out.device
    with torch.cuda.device(device):
        # Calculate the total number of random numbers to generate
        N = out.numel()

        # Calculate the grid size
        BLOCK = 128  # Block size
        num_warps = 4  # Number of warps per block
        grid = (N + BLOCK - 1) // BLOCK  # Number of blocks

        # Initialize the Philox generator state
        philox_cuda_seed_offset = triton_runtime.philox_cuda_seed_offset(philox_seed, philox_offset)

        # Launch the kernel
        uniform_kernel[grid, BLOCK, num_warps](
            out_ptr=out.data_ptr(),
            N=N,
            philox_seed=philox_cuda_seed_offset.seed,
            philox_offset=philox_cuda_seed_offset.offset,
            from_=from_,
            to=to,
            BLOCK=BLOCK
        )

# Example usage
if __name__ == "__main__":
    # Define the output tensor
    out = torch.empty(1024, device="cuda")

    # Define the range and Philox parameters
    from_ = 0.0
    to = 1.0
    philox_seed = 12345
    philox_offset = 0

    # Generate uniform random numbers
    uniform_(out, from_, to, philox_seed, philox_offset)

    # Print the result
    print(out)
