import torch
import triton
import triton.language as tl

@triton.jit
def rsqrt_kernel(input_ptr, output_ptr, n_elements,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.math.rsqrt(x)
    tl.store(output_ptr + offsets, y, mask=mask)

def rsqrt(input: torch.Tensor, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Shape of out must be the same as input"
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    rsqrt_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
