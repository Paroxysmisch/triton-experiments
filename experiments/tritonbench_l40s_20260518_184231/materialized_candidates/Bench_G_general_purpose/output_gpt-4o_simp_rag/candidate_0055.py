import torch
import triton
import triton.language as tl

BLOCK = 1024

# Kernel for generating uniform random numbers within a specified range
@triton.jit
def uniform_kernel(out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr):
    # Calculate the range width
    range_width = to - from_
    
    # Calculate offsets for each thread
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    
    # Initialize Philox PRNG with the given seed and offset
    key = tl.make_uint2(philox_seed, philox_offset)
    rand_uint = tl.philox(key, offset)
    
    # Convert the random uint32 to float and scale to [0, 1)
    rand_float = tl.random.uint32_to_uniform_float(rand_uint)
    
    # Scale and shift the random numbers to the desired range [from_, to)
    rand_scaled = rand_float * range_width + from_
    
    # Store the result in the output pointer
    tl.store(out_ptr + offset, rand_scaled, mask=offset < N)

# Wrapper function to launch the uniform_kernel
def generate_uniform_random_numbers(size, philox_seed, philox_offset, from_, to, device='cuda'):
    # Prepare output tensor
    x = torch.empty(size, dtype=torch.float32, device=device)
    N = x.numel()
    
    # Define the grid size
    grid = (triton.cdiv(N, BLOCK),)
    
    # Launch the kernel
    uniform_kernel[grid](x, N, philox_seed, philox_offset, from_, to, BLOCK=BLOCK)
    
    return x

# Example usage
if __name__ == "__main__":
    size = (4096,)  # Example size
    philox_seed = 12345  # Example seed
    philox_offset = 0  # Example offset
    from_ = 0.0  # Start of the range
    to = 1.0  # End of the range

    # Generate uniform random numbers
    random_numbers = generate_uniform_random_numbers(size, philox_seed, philox_offset, from_, to)
    print(random_numbers)
