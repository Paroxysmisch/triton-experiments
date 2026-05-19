import triton
import triton.language as tl

@triton.jit
def sqrt_tanh_kernel(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr  # Block size
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_block = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute sqrt and then tanh
    sqrt_block = tl.sqrt(input_block)
    tanh_block = tl.tanh(sqrt_block)
    
    tl.store(output_ptr + offsets, tanh_block, mask=mask)

import torch
import triton
import triton.language as tl

def sqrt_tanh(input, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    sqrt_tanh_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    
    return out
