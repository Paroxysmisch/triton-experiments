import triton
import triton.language as tl

@triton.jit
def chunk_global_cumsum_vector_kernel(
    s_ptr,  # Pointer to the input tensor
    s_cumsum_ptr,  # Pointer to the output tensor
    N,  # Total number of elements in the tensor
    chunk_size,  # Size of each chunk
    stride,  # Stride along the specified dimension
    BLOCK_SIZE: tl.constexpr,  # Block size for the kernel
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load the block of data
    mask = offsets < N
    s_block = tl.load(s_ptr + offsets, mask=mask, other=0.0)

    # Compute the lower triangular mask
    lower_tri_mask = tl.arange(0, BLOCK_SIZE)[:, None] <= tl.arange(0, BLOCK_SIZE)[None, :]

    # Compute the cumulative sum using matrix multiplication
    s_cumsum_block = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        s_cumsum_block += tl.where(lower_tri_mask[i, :], s_block, 0.0)

    # Store the result back
    tl.store(s_cumsum_ptr + offsets, s_cumsum_block, mask=mask)

import torch

def chunk_global_cumsum_vector(s, chunk_size, dim=0):
    N = s.numel()
    s_cumsum = torch.zeros_like(s)

    # Define the grid and block sizes
    BLOCK_SIZE = 128  # This can be tuned for better performance
    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    chunk_global_cumsum_vector_kernel[grid, BLOCK_SIZE](
        s,  # Input tensor
        s_cumsum,  # Output tensor
        N,  # Total number of elements
        chunk_size,  # Size of each chunk
        s.stride(dim),  # Stride along the specified dimension
        BLOCK_SIZE,  # Block size
    )

    return s_cumsum
