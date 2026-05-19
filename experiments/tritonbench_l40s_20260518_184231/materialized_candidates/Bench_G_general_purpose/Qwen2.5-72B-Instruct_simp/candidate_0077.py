import triton
import triton.language as tl

@triton.jit
def _seeded_dropout(X, Y, p, seed, BLOCK_SIZE: tl.constexpr):
    # Get the block index and the block size
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Get the block of data
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < X.shape[0]
    x = tl.load(X + offsets, mask=mask)

    # Generate random numbers
    rng = tl.rand(seed, offsets)
    dropout_mask = rng > p

    # Apply dropout mask and scale
    y = tl.where(dropout_mask, x / (1 - p), 0.0)

    # Write back the result
    tl.store(Y + offsets, y, mask=mask)

import torch
import triton
import triton.language as tl

def seeded_dropout(x: torch.Tensor, p: float, seed: int, block_size: int = 1024):
    # Ensure the input tensor is on the GPU
    assert x.is_cuda, "Input tensor must be on the GPU"
    
    # Create the output tensor with the same shape as the input
    y = torch.empty_like(x)
    
    # Get the grid size
    grid = (triton.cdiv(x.numel(), block_size),)
    
    # Launch the kernel
    _seeded_dropout[grid](x, y, p, seed, BLOCK_SIZE=block_size)
    
    return y
