import torch
import triton
import triton.language as tl

@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Generate random values and create dropout mask
    random = tl.rand(seed, offsets)
    x_keep = random > p
    
    # Apply dropout and scaling
    output = tl.where(x_keep, x / (1 - p), 0.0)
    
    # Write-back
    tl.store(output_ptr + offsets, output, mask=mask)

def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous(), "Input tensor must be contiguous"
    n_elements = x.numel()
    
    # Define grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    
    return output

# Example usage
if __name__ == "__main__":
    x = torch.randn(size=(10,)).cuda()
    output1 = seeded_dropout(x, p=0.5, seed=123)
    output2 = seeded_dropout(x, p=0.5, seed=123)
    output3 = seeded_dropout(x, p=0.5, seed=512)
    
    print("Input:", x)
    print("Output 1 (seed=123):", output1)
    print("Output 2 (seed=123):", output2)
    print("Output 3 (seed=512):", output3)
