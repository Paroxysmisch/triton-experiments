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
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    x_keep = x_keep != 0  # Convert to boolean mask
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous(), "Input x must be contiguous."
    assert x_keep.is_contiguous(), "Mask x_keep must be contiguous."
    assert x.shape == x_keep.shape, "x and x_keep must have the same shape."
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output

# Example usage
x = torch.randn(10, device='cuda')
x_keep = torch.rand(10, device='cuda') > 0.5  # Random boolean mask
output = dropout(x, x_keep, p=0.5)
