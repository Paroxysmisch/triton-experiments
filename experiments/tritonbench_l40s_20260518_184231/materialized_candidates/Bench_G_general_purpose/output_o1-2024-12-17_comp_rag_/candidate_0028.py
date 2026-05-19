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
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    y = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, y, mask=mask)

def dropout(x, x_keep, p):
    assert x.is_contiguous()
    assert x_keep.is_contiguous()
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output

@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    r = tl.rand(seed, offsets)
    x_keep = r > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def seeded_dropout(x, p, seed):
    assert x.is_contiguous()
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output

# Example usage:
if __name__ == "__main__":
    x = torch.randn(size=(10,)).cuda()
    x_keep = (torch.rand_like(x) > 0.5).bool()
    out = dropout(x, x_keep, p=0.5)

    out_seeded_1 = seeded_dropout(x, p=0.5, seed=123)
    out_seeded_2 = seeded_dropout(x, p=0.5, seed=123)
    out_seeded_3 = seeded_dropout(x, p=0.5, seed=512)
