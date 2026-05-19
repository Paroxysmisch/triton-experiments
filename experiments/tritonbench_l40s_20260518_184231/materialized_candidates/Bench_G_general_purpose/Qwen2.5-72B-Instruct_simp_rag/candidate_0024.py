import torch
import triton
import triton.language as tl

# Triton kernel for seeded dropout
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
    
    # Generate random numbers and apply the mask
    random = tl.rand(seed, offsets)
    x_keep = random > p
    
    # Scale and store the result
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

# Wrapper function for seeded dropout
def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output

# Example usage
x = torch.randn(size=(10,)).cuda()
output = seeded_dropout(x, p=0.5, seed=123)
output2 = seeded_dropout(x, p=0.5, seed=123)
output3 = seeded_dropout(x, p=0.5, seed=512)

print("Input tensor:")
print(x)
print("Output tensor with seed 123:")
print(output)
print("Output tensor with seed 123 (again):")
print(output2)
print("Output tensor with seed 512:")
print(output3)
