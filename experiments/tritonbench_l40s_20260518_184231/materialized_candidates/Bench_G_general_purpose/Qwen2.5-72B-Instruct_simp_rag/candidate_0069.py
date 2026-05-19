import triton
import triton.language as tl

# Kernel function for seeded dropout.
@triton.jit
def _seeded_dropout(
    x_ptr,  # pointer to the input
    output_ptr,  # pointer to the output
    n_elements,  # number of elements in the `x` tensor
    p,  # dropout probability
    seed,  # random seed
    BLOCK_SIZE: tl.constexpr,
):
    # Compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Generate random numbers and apply dropout
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # Write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

# Python wrapper function for the seeded dropout kernel
def seeded_dropout(x, p, seed, BLOCK_SIZE=1024):
    # Ensure x is a Triton tensor
    if not isinstance(x, tl.tensor):
        x = tl.tensor(x, dtype=tl.float32)
    
    # Create output tensor with the same shape as x
    output = tl.empty_like(x)
    
    # Launch the kernel
    grid = (tl.cdiv(x.size, BLOCK_SIZE),)
    _seeded_dropout[grid](x, output, x.size, p, seed, BLOCK_SIZE)
    
    return output

# Example usage
import torch

# Create a tensor
x = torch.randn(1024, device='cuda')

# Apply seeded dropout
p = 0.5
seed = 42
output = seeded_dropout(x, p, seed)

# Print the output
print(output)
