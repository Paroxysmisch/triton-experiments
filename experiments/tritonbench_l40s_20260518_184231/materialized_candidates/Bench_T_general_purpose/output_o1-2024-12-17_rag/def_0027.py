import triton
import triton.language as tl
import torch
import math

@triton.jit
def _sqrt_tanh_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    val = tl.load(a_ptr + offset, mask=mask)
    out_val = tl.tanh(tl.sqrt(val.to(tl.float32)))
    tl.store(b_ptr + offset, out_val, mask=mask)

def sqrt_tanh(input, out=None):
    if out is None:
        out = torch.empty_like(input)
    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    _sqrt_tanh_kernel[(grid_size,)](input, out, n_elements, block_size)
    return out
