import torch
import triton
import triton.language as tl

@triton.jit
def _dropout(
    x_ptr,
    x_keep_ptr,
    output_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data from x and x_keep
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    
    # Apply dropout
    output = tl.where(x_keep, x / (1 - p), 0.0)
    
    # Write-back
    tl.store(output_ptr + offsets, output, mask=mask)

def dropout(x, x_keep, p):
    assert x.is_contiguous()
    assert x_keep.is_contiguous()
    assert x.shape == x_keep.shape
    
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    _dropout[grid](
        x, x_keep, output, n_elements, p, BLOCK_SIZE=1024
    )
    
    return output

# Example usage
x = torch.randn(size=(10,)).cuda()
x_keep = torch.randint(0, 2, size=(10,), dtype=torch.bool).cuda()
output = dropout(x, x_keep, p=0.5)
