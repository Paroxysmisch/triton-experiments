import triton
import triton.language as tl
import torch

# Kernel function for seeded dropout
@triton.jit
def _seeded_dropout(
    x_ptr,        # pointer to the input tensor
    output_ptr,   # pointer to the output tensor
    n_elements,   # number of elements in the input tensor
    p,            # dropout probability
    seed,         # seed for random number generation
    BLOCK_SIZE: tl.constexpr,  # block size
):
    # Compute memory offsets for the current block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't access out-of-bounds memory
    mask = offsets < n_elements
    
    # Load data from the input tensor
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Generate a random mask for dropout using the seed
    random = tl.rand(seed, offsets)
    x_keep = random > p
    
    # Apply dropout: scale kept elements, zero out others
    output = tl.where(x_keep, x / (1 - p), 0.0)
    
    # Write the result to the output tensor
    tl.store(output_ptr + offsets, output, mask=mask)

# Python wrapper function for the seeded dropout kernel
def seeded_dropout(x, p, seed, block_size=1024):
    # Ensure input is a torch tensor and get the number of elements
    assert isinstance(x, torch.Tensor), "Input must be a torch tensor"
    n_elements = x.numel()
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Launch the Triton kernel
    grid = (triton.cdiv(n_elements, block_size),)
    _seeded_dropout[grid](
        x_ptr=x, 
        output_ptr=output, 
        n_elements=n_elements, 
        p=p, 
        seed=seed, 
        BLOCK_SIZE=block_size
    )
    
    return output
