import triton
import triton.language as tl
import torch

@triton.jit
def _selu_kernel(in_ptr, out_ptr, n_elements, alpha, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    y = scale * (tl.maximum(x, 0.) + tl.minimum(0., alpha * (tl.exp(x) - 1.)))
    tl.store(out_ptr + offsets, y, mask=mask)

def selu(input, inplace=False):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    if inplace:
        out = input
    else:
        out = torch.empty_like(input)
    n_elements = input.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _selu_kernel[grid](input, out, n_elements, alpha, scale, BLOCK_SIZE=1024)
    return out
