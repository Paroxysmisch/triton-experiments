import torch
import triton
import triton.language as tl


@triton.jit
def uniform_kernel(
    out_ptr,  # Pointer to the output buffer
    N,  # Total number of random numbers to generate
    philox_seed,  # Philox RNG seed
    philox_offset,  # Philox RNG offset
    from_,  # Lower bound of the range
    to,  # Upper bound of the range
    BLOCK: tl.constexpr,  # Number of threads per block
):
    # Programmatically determine the current block's start index
    pid = tl.program_id(0)
    start_idx = pid * BLOCK + tl.arange(0, BLOCK)

    # Calculate the Philox counter and offset for this block
    counter = philox_offset + start_idx // 4
    thread_offset = start_idx % 4

    # Loop through each element in the block
    for i in range(0, BLOCK, 4):
        # Generate 4 random uints using Philox
        r0, r1, r2, r3 = tl.philox(philox_seed, counter + i // 4)

        # Convert uints to floats in [0, 1)
        f0 = tl.uint_to_uniform_float(r0)
        f1 = tl.uint_to_uniform_float(r1)
        f2 = tl.uint_to_uniform_float(r2)
        f3 = tl.uint_to_uniform_float(r3)

        # Scale to the range [from_, to)
        f0 = from_ + f0 * (to - from_)
        f1 = from_ + f1 * (to - from_)
        f2 = from_ + f2 * (to - from_)
        f3 = from_ + f3 * (to - from_)

        # Calculate offsets for storing the random numbers
        off_0 = start_idx + i + 0
        off_1 = start_idx + i + 1
        off_2 = start_idx + i + 2
        off_3 = start_idx + i + 3

        # Conditionally store random numbers based on offset validity
        if off_0 < N:
            tl.store(out_ptr + off_0, f0, eviction_policy="evict_first")
        if off_1 < N:
            tl.store(out_ptr + off_1, f1, eviction_policy="evict_first")
        if off_2 < N:
            tl.store(out_ptr + off_2, f2, eviction_policy="evict_first")
        if off_3 < N:
            tl.store(out_ptr + off_3, f3, eviction_policy="evict_first")


def uniform_(out, N, philox_seed, philox_offset, from_, to):
    """
    High-level wrapper for generating uniform random numbers using the uniform_kernel.

    Args:
        out (torch.Tensor): Output tensor to store the random numbers.
        N (int): Total number of random numbers to generate.
        philox_seed (int): Seed for the Philox RNG.
        philox_offset (int): Offset for the Philox RNG.
        from_ (float): Lower bound of the random number range.
        to (float): Upper bound of the random number range.
    """
    # Ensure the output tensor is on the correct device
    assert out.is_cuda, "Output tensor must be on a CUDA device."

    # Heuristics to determine the block size and number of warps
    BLOCK = 256  # Default block size
    num_warps = 4  # Default number of warps

    # Calculate the number of blocks required
    grid = (N + BLOCK - 1) // BLOCK

    # Launch the Triton kernel
    uniform_kernel[grid](
        out_ptr=out.data_ptr(),
        N=N,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        from_=from_,
        to=to,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )


# Example Usage
if __name__ == "__main__":
    # Parameters
    N = 1024  # Number of random numbers
    from_ = 0.0  # Lower bound
    to = 1.0  # Upper bound
    philox_seed = 12345  # RNG seed
    philox_offset = 0  # RNG offset

    # Output tensor
    out = torch.empty(N, device="cuda", dtype=torch.float32)

    # Generate random numbers
    uniform_(out, N, philox_seed, philox_offset, from_, to)

    # Print a few random numbers
    print(out[:10])
