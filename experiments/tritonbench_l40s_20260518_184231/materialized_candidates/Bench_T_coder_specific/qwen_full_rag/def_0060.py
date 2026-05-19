import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume

# Triton kernel to compute exp(sqrt(x))
@triton.jit
def _exp_sqrt(x, y, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_row = tl.load(x + offsets, mask=mask).to(tl.float32)
    x_row = tl.exp(x_row)
    x_row = tl.sqrt(x_row)
    tl.store(y + offsets, x_row, mask=mask)

# Wrapper function to call the Triton kernel
def exp_sqrt(input, out=None):
    if out == None:
        out = torch.empty_like(input)
    n_elements = volume(input.shape)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _exp_sqrt[grid](input, out, n_elements, BLOCK_SIZE=1024)
    return out
