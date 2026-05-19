import torch
import triton
import triton.language as tl

@triton.jit
def sqrt_exp_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_ptrs = input + offsets
    output_ptrs = output + offsets
    x = tl.load(input_ptrs, mask=mask)
    output_val = tl.exp(tl.sqrt(x))
    tl.store(output_ptrs, output_val, mask=mask)

def sqrt_exp(input, out=None) -> torch.Tensor:
    assert input.is_contiguous()
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_contiguous()
    assert out.shape == input.shape
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sqrt_exp_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
