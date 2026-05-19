import triton
import triton.language as tl
import torch

@triton.jit
def _seeded_dropout(x_ptr, output_ptr, p, seed, BLOCK_SIZE, n_elements, **meta):
    # Compute the index of the block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load the data
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements, other=0.0)
    
    # Initialize random state
    rng_state = tl.program_id(0) + seed
    # Generate random numbers for dropout mask
    random_values = tl.rand(rng_state, offsets)
    
    # Create dropout mask
    mask = random_values > p
    
    # Apply mask and scale the output
    output = tl.where(mask, x / (1 - p), 0.0)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=offsets < n_elements)

def seeded_dropout(x, p, seed, BLOCK_SIZE=1024):
    assert 0 <= p <= 1, "Dropout probability must be in [0, 1]"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Get the number of elements in the input tensor
    n_elements = x.numel()
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _seeded_dropout[grid](
        x, output, p, seed, BLOCK_SIZE, n_elements,
        num_warps=4  # Number of warps to use for the kernel
    )
    
    return output

# Example usage
x = torch.randn(4096, device='cuda')
p = 0.5
seed = 42
output = seeded_dropout(x, p, seed)
