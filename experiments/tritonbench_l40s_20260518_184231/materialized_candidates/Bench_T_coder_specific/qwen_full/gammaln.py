import torch
import triton
import triton.language as tl

@triton.jit
def _gammaln_kernel(input, output, n_elements,
                    BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_ptrs = input + offsets
    output_ptrs = output + offsets
    x = tl.load(input_ptrs, mask=mask)
    x = tl.math.gammaln(tl.abs(x))
    tl.store(output_ptrs, x, mask=mask)

def gammaln(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        out = out.contiguous()
    assert input.is_contiguous()
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _gammaln_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
