import torch
import triton
import triton.language as tl

@triton.jit
def exp_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_ptrs = input + offsets
    output_ptrs = output + offsets
    x = tl.load(input_ptrs, mask=mask)
    output = tl.exp(x)
    tl.store(output_ptrs, output, mask=mask)

def exp(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    assert input.is_contiguous()
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    exp_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
