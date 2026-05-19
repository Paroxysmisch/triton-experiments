import torch
import triton
import triton.language as tl

@triton.jit
def q_kernel_per_block_int8(
    q: tl.tensor,  # [BLKQ, ..., D]
    q_int8: tl.tensor,  # [BLKQ, ..., D]
    q_scale: tl.tensor,  # [BLKQ, ..., 1]
    BLKQ: tl.int32,
    D: tl.int32
):
    # Get program ID and coordinates
    pid = tl.program_id(0)
    coords = tl.program_id(1, 2)

    # Calculate offsets
    block_offset = pid * BLKQ
    q_offset = block_offset * D
    q_int8_offset = block_offset * D
    q_scale_offset = block_offset

    # Load block of query matrix
    q_block = tl.load(q + q_offset, mask=coords < D, other=0.0)

    # Compute max absolute value in the block
    q_max_abs = tl.max(tl.abs(q_block))

    # Compute scaling factor
    scale = 127.0 / q_max_abs

    # Quantize to int8 with nearest integer rounding
    q_int8_block = tl.round(q_block * scale)

    # Store quantized data and scaling factor
    tl.store(q_int8 + q_int8_offset, q_int8_block, mask=coords < D)
    tl.store(q_scale + q_scale_offset, scale)

@triton.jit
def k_kernel_per_block_int8(
    k: tl.tensor,  # [BLKK, ..., D]
    k_int8: tl.tensor,  # [BLKK, ..., D]
    k_scale: tl.tensor,  # [BLKK, ..., 1]
    BLKK: tl.int32,
    D: tl.int32
):
    # Get program ID and coordinates
    pid = tl.program_id(0)
    coords = tl.program_id(1, 2)

    # Calculate offsets
    block_offset = pid * BLKK
    k_offset = block_offset * D
    k_int8_offset = block_offset * D
    k_scale_offset = block_offset

    # Load block of key matrix
    k_block = tl.load(k + k_offset, mask=coords < D, other=0.0)

    # Compute max absolute value in the block
    k_max_abs = tl.max(tl.abs(k_block))

    # Compute scaling factor
    scale = 127.0 / k_max_abs

    # Quantize to int8 with nearest integer rounding
    k_int8_block = tl.round(k_block * scale)

    # Store quantized data and scaling factor
    tl.store(k_int8 + k_int8_offset, k_int8_block, mask=coords < D)
    tl.store(k_scale + k_scale_offset, scale)

def per_block_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    BLKQ: int,
    BLKK: int,
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor
):
    # Reshape input tensors for uniform handling
    q = q.contiguous().view(-1, BLKQ, q.shape[-1])
    k = k.contiguous().view(-1, BLKK, k.shape[-1])

    # Initialize output tensors
    q_int8 = q_int8.contiguous().view(-1, BLKQ, q_int8.shape[-1])
    k_int8 = k_int8.contiguous().view(-1, BLKK, k_int8.shape[-1])
    q_scale = q_scale.contiguous().view(-1, BLKQ, 1)
    k_scale = k_scale.contiguous().view(-1, BLKK, 1)

    # Calculate grid configuration
    grid = (q.shape[0], q.shape[1], 1)
    block = (BLKQ, 1, 1)

    # Launch Triton kernels
    q_kernel_per_block_int8[grid, block](q, q_int8, q_scale, BLKQ, q.shape[-1])
    k_kernel_per_block_int8[grid, block](k, k_int8, k_scale, BLKK, k.shape[-1])

    # Reshape output tensors back to original dimensions
    q_int8 = q_int8.view(q.shape[0], -1)
    k_int8 = k_int8.view(k.shape[0], -1)
    q_scale = q_scale.view(q.shape[0], -1)
    k_scale = k_scale.view(k.shape[0], -1)

    return q_int8, k_int8, q_scale, k_scale
