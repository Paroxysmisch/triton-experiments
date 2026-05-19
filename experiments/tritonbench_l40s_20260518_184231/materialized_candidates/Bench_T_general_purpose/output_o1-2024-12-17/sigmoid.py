import triton
import triton.language as tl
import torch

@triton.jit
def _sigmoid_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)

def sigmoid(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _sigmoid_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
