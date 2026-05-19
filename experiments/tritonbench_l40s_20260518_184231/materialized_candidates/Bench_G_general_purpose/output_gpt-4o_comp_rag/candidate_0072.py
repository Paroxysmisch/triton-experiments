import triton
import triton.language as tl
import torch

# Kernel function for seeded dropout.
@triton.jit
def _seeded_dropout(
    x_ptr,  # pointer to the input tensor
    output_ptr,  # pointer to the output tensor
    n_elements,  # number of elements in the input tensor
    p,  # dropout probability
    seed,  # random seed for reproducibility
    BLOCK_SIZE: tl.constexpr,  # size of the block of data processed by each kernel instance
):
    # Compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't read/write out of bounds
    mask = offsets < n_elements
    
    # Load data from x
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Generate random numbers for dropout
    random = tl.rand(seed, offsets)
    
    # Create dropout mask
    x_keep = random > p
    
    # Apply dropout and scale the kept elements
    output = tl.where(x_keep, x / (1 - p), 0.0)
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def seeded_dropout(x, p, seed, BLOCK_SIZE=1024):
    # Ensure input tensor is contiguous
    x = x.contiguous()
    
    # Number of elements in the input tensor
    n_elements = x.numel()
    
    # Create output tensor of the same shape and type as input tensor
    output = torch.empty_like(x)
    
    # Calculate grid size to cover all elements
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the Triton kernel
    _seeded_dropout[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        p=p,
        seed=seed,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Example usage:
x = torch.randn(10000, device='cuda')
p = 0.5
seed = 42
output = seeded_dropout(x, p, seed)
print(output)
