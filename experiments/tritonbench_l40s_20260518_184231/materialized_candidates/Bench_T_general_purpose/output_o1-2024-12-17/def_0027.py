import triton
import triton.language as tl
import torch

@triton.jit
def _sqrt_tanh_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    x_sqrt = tl.sqrt(x)
    x_tanh = tl.tanh(x_sqrt)
    tl.store(out_ptr + offsets, x_tanh, mask=mask)

def sqrt_tanh(input, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _sqrt_tanh_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
