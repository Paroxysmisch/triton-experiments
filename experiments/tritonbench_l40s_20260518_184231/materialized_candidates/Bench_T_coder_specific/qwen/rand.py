import triton
import triton.language as tl

@triton.jit
def rand_kernel(
    out_ptr,
    size,
    num_elements,
    block_size=256,
):
    # Each block processes 'block_size' elements
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)

    # Generate random numbers using the Mersenne Twister algorithm
    seed = pid
    random_numbers = tl.rand(seed, offsets, num_elements)

    # Store the results in the output tensor
    out_ptr[tl.arange(block_start, min(block_start + block_size, num_elements))] = random_numbers
