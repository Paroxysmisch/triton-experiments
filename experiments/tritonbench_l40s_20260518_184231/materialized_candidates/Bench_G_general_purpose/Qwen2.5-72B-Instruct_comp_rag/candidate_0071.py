import triton
import triton.language as tl

# Kernel function for dropout using a precomputed mask.
@triton.jit
def _triton_dropout(
    x_ptr,  # pointer to the input
    x_keep_ptr,  # pointer to a mask of 0s and 1s
    output_ptr,  # pointer to the output
    n_elements,  # number of elements in the `x` tensor
    p,  # probability that an element of `x` is changed to zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # Perform dropout
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)

# Kernel function for seeded dropout.
@triton.jit
def _seeded_triton_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # Write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

import torch

def triton_dropout(x, x_keep, p, BLOCK_SIZE=1024):
    # Ensure the input tensor is contiguous
    x = x.contiguous()
    x_keep = x_keep.contiguous()
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the kernel
    _triton_dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE)
    
    return output

def seeded_triton_dropout(x, p, seed, BLOCK_SIZE=1024):
    # Ensure the input tensor is contiguous
    x = x.contiguous()
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the kernel
    _seeded_triton_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE)
    
    return output
