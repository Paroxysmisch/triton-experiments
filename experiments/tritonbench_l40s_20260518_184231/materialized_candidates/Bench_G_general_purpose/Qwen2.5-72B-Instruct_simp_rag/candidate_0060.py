import torch
import triton
import triton.language as tl

BLOCK = 1024

# Triton kernel for generating uniform random numbers
@triton.jit
def uniform_kernel(out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < N

    # Generate 4 random numbers per thread using Philox
    rand4 = tl.philox4x(philox_seed, philox_offset + offsets)
    rand4 = (rand4 * (to - from_) / (2**32)) + from_  # Convert to [from_, to) range

    # Store the random numbers to memory
    tl.store(out_ptr + offsets, rand4, mask=mask)

# Wrapper function to launch the kernel
def generate_uniform_random_numbers(out_tensor, philox_seed, philox_offset, from_, to, device='cuda'):
    N = out_tensor.numel()
    grid = (triton.cdiv(N, BLOCK),)
    uniform_kernel[grid](out_tensor, N, philox_seed, philox_offset, from_, to, BLOCK)

# Example usage
if __name__ == "__main__":
    N = 1024 * 1024  # Number of random numbers to generate
    from_ = 0.0
    to = 1.0
    philox_seed = 42
    philox_offset = 0

    out_tensor = torch.empty(N, dtype=torch.float32, device='cuda')
    generate_uniform_random_numbers(out_tensor, philox_seed, philox_offset, from_, to)

    # Verify the generated random numbers
    out_np = out_tensor.cpu().numpy()
    assert np.all((out_np >= from_) & (out_np < to))
    print("Random numbers generated successfully and are within the specified range.")
