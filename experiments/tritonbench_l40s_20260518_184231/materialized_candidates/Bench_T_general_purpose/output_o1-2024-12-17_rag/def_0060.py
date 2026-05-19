import triton
import triton.language as tl
import torch
import math

@triton.jit
def exp_sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(input_ptr + offset, mask=mask)
    # Compute sqrt(exp(x))
    y = tl.exp(x.to(tl.float32))
    y = tl.sqrt(y)
    tl.store(output_ptr + offset, y, mask=mask)

def exp_sqrt(input, out=None):
    if out is None:
        out = torch.empty_like(input)
    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid = triton.cdiv(n_elements, block_size)
    exp_sqrt_kernel[(grid,)](input, out, n_elements, block_size)
    return out
