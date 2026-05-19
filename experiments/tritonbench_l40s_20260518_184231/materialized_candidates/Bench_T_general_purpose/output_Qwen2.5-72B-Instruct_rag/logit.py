import torch
import triton
import triton.language as tl

@triton.jit
def logit_kernel(output_ptr, input_ptr, n_elements, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_values = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    if eps is not None:
        eps = tl.float32(eps)
        input_values = tl.where(input_values < eps, eps, input_values)
        input_values = tl.where(input_values > (1.0 - eps), 1.0 - eps, input_values)
    else:
        input_values = tl.where((input_values < 0.0) | (input_values > 1.0), tl.nan, input_values)

    logits = tl.log(input_values / (1.0 - input_values))
    tl.store(output_ptr + offsets, logits, mask=mask)

import torch

def logit(input, eps=None, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024  # Adjust block size as needed

    logit_kernel[(n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, ](
        out, input, n_elements, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
