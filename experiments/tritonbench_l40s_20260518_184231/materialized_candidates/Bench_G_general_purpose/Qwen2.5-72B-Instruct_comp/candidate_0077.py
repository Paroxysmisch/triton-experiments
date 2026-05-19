import triton
import triton.language as tl

@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    rng = tl.rand(seed, offsets)
    keep_mask = rng > p
    output = tl.where(keep_mask, x / (1 - p), 0.0)

    tl.store(output_ptr + offsets, output, mask=mask)

import torch
import triton
import triton.language as tl

def seeded_dropout(x, p, seed):
    # Ensure the input tensor is contiguous
    x = x.contiguous()
    
    # Initialize the output tensor with the same shape and type as the input tensor
    output = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    
    # Define the block size (number of elements processed by each kernel instance)
    BLOCK_SIZE = 1024
    
    # Calculate the grid size (number of blocks needed to cover all elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the Triton kernel
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE)
    
    return output
