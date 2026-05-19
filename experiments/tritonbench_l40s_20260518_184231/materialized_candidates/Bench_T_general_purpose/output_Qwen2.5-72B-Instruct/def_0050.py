import triton
import triton.language as tl

@triton.jit
def sqrt_exp_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_block = tl.load(input_ptr + offsets, mask=mask)
    sqrt_block = tl.sqrt(input_block)
    exp_block = tl.exp(sqrt_block)
    tl.store(output_ptr + offsets, exp_block, mask=mask)

import torch
import triton
import triton.language as tl

def sqrt_exp(input, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    
    sqrt_exp_kernel[grid](
        input, out, n_elements, BLOCK_SIZE=1024
    )
    
    return out
