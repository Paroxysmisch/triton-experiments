import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    s_ptr,  # Pointer to the input tensor
    o_ptr,  # Pointer to the output tensor
    B,      # Batch size
    H,      # Number of heads
    T,      # Sequence length
    BT,     # Block size for the T dimension
    stride_s_b,  # Stride for the batch dimension in the input tensor
    stride_s_h,  # Stride for the head dimension in the input tensor
    stride_s_t,  # Stride for the sequence dimension in the input tensor
    stride_o_b,  # Stride for the batch dimension in the output tensor
    stride_o_h,  # Stride for the head dimension in the output tensor
    stride_o_t,  # Stride for the sequence dimension in the output tensor
    BLOCK_T: tl.constexpr,  # Block size for the T dimension
):
    # Get the current (B, H) pair
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Initialize the accumulation variable
    b_z = tl.zeros((1,), dtype=tl.float32)

    # Iterate over the T dimension in blocks of size BT, moving backwards
    for block_start in range(T - 1, -1, -BLOCK_T):
        block_end = max(block_start - BLOCK_T + 1, 0)
        block_size = block_start - block_end + 1

        # Load the current block into a Triton block
        block = tl.load(
            s_ptr + b * stride_s_b + h * stride_s_h + block_start * stride_s_t,
            mask=block_start - tl.arange(0, BLOCK_T) >= block_end,
            other=0.0
        )

        # Compute the sum of the current block
        block_sum = tl.sum(block, axis=0)

        # Update the accumulation variable
        b_z += block_sum

        # Calculate the cumulative sum and store the result
        result = b_z - block
        tl.store(
            o_ptr + b * stride_o_b + h * stride_o_h + block_start * stride_o_t,
            result,
            mask=block_start - tl.arange(0, BLOCK_T) >= block_end
        )

import torch

def chunk_global_reversed_cumsum_scalar(s: torch.Tensor) -> torch.Tensor:
    B, H, T = s.shape
    BLOCK_T = 128  # Block size for the T dimension

    # Ensure the input tensor is on the GPU
    s = s.cuda()

    # Allocate the output tensor
    o = torch.empty_like(s, device=s.device)

    # Define the grid and block sizes
    grid = (B * H,)
    block = (BLOCK_T,)

    # Launch the kernel
    chunk_global_reversed_cumsum_scalar_kernel[grid, block](
        s, o, B, H, T, BLOCK_T,
        s.stride(0), s.stride(1), s.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_T
    )

    return o
