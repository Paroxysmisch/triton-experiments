import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    s_ptr,  # Pointer to the input tensor
    o_ptr,  # Pointer to the output tensor
    B,      # Batch size
    H,      # Number of heads
    T,      # Sequence length
    BT,     # Chunk size
    BLOCK_SIZE: tl.constexpr
):
    # Get the batch and head indices
    pid = tl.program_id(0)
    bid = pid // H
    hid = pid % H

    # Compute the starting and ending indices for the chunk
    chunk_start = T - (pid % (T // BT)) * BT
    chunk_end = max(chunk_start - BT, 0)

    # Compute the offset for the current (B, H) block
    offset = bid * H * T + hid * T

    # Initialize the cumulative sum
    cumsum = 0.0

    # Iterate over the chunk in reverse order
    for t in range(chunk_start - 1, chunk_end - 1, -1):
        # Compute the global index
        idx = offset + t

        # Load the value from the input tensor
        s_val = tl.load(s_ptr + idx)

        # Update the cumulative sum
        cumsum += s_val

        # Store the result in the output tensor
        tl.store(o_ptr + idx, cumsum)

import torch

def chunk_global_reversed_cumsum_scalar(s, BT):
    # Get the dimensions of the input tensor
    B, H, T = s.shape

    # Initialize the output tensor
    o = torch.zeros_like(s)

    # Define the grid and block sizes
    grid = (B * H, 1, 1)
    block = (1, 1, 1)

    # Launch the kernel
    chunk_global_reversed_cumsum_scalar_kernel[grid, block](
        s, o, B, H, T, BT, BLOCK_SIZE=1
    )

    return o
