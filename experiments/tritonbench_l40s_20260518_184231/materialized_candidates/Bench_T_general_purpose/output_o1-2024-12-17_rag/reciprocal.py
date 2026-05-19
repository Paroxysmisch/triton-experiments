import triton
import triton.language as tl
import torch
import math

@triton.jit
def reciprocal_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    in_val = tl.load(input_ptr + offset, mask=mask)
    out_val = 1 / in_val.to(tl.float32)
    tl.store(output_ptr + offset, out_val, mask=mask)

def reciprocal(input, *, out=None):
    if not input.is_floating_point():
        input = input.to(torch.get_default_dtype())
    if out is None:
        out = torch.empty_like(input)
    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid = triton.cdiv(n_elements, block_size)
    reciprocal_kernel[(grid,)](
        input, out, n_elements, block_size
    )
    return out
