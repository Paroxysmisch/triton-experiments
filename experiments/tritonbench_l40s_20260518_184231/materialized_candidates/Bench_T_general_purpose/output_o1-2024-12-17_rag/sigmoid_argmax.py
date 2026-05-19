import torch
import triton
import triton.language as tl

@triton.jit
def _apply_sigmoid_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x = 1.0 / (1.0 + tl.exp(-x))  # sigmoid
    tl.store(output_ptr + offsets, x, mask=mask)

def sigmoid_argmax(input, dim=None, keepdim=False):
    x = input.contiguous()
    n_elements = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _apply_sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out.argmax(dim=dim, keepdim=keepdim)
