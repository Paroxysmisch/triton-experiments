import triton
import triton.language as tl
import torch

@triton.jit
def _exp_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    y = tl.exp(x)
    tl.store(out_ptr + offsets, y, mask=mask)

def exp(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    assert input.is_contiguous()
    assert out.is_contiguous()
    n_elements = input.numel()
    in_ptr = input.data_ptr()
    out_ptr = out.data_ptr()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _exp_kernel[grid](in_ptr, out_ptr, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
