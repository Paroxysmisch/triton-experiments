import triton
import triton.language as tl

@triton.jit
def logit_kernel(input_ptr, output_ptr, n_elements, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_block = tl.load(input_ptr + offsets, mask=mask)
    
    if eps is not None:
        input_block = tl.where(input_block < eps, eps, input_block)
        input_block = tl.where(input_block > 1 - eps, 1 - eps, input_block)
    else:
        input_block = tl.where((input_block < 0) | (input_block > 1), tl.nan, input_block)

    output_block = tl.log(input_block / (1 - input_block))
    tl.store(output_ptr + offsets, output_block, mask=mask)

import torch
import triton
import triton.language as tl

def logit(input, eps=None, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    
    logit_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=out.data_ptr(),
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=1024
    )
    
    return out

import torch

# Example input tensor
input_tensor = torch.tensor([0.1, 0.5, 0.9, -0.1, 1.1], dtype=torch.float32)

# Call the logit function
output_tensor = logit(input_tensor, eps=1e-6)

print(output_tensor)
