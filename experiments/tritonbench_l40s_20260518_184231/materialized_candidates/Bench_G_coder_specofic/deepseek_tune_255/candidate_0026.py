import torch
import triton
import triton.language as tl

# Triton kernel for converting query matrices to int8
@triton.jit
def q_kernel_per_block_int8(
    q, q_int8, q_scale, BLKQ, BLOCK: tl.constexpr,
):
    # Determine offsets for current program instance
    program_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offset_q = program_id * BLKQ * BLKQ * 16 + block_id * BLKQ * 16
    offset_q_int8 = program_id * BLKQ * BLKQ + block_id * BLKQ
    offset_q_scale = program_id * BLKQ * BLKQ + block_id * BLKQ

    # Load block of query matrix
    q_block = tl.load(
        q + offset_q + tl.arange(0, 16)[:, None] * BLKQ + tl.arange(0, 16)[None, :]
    )

    # Compute scaling factor for normalization
    scale = tl.max(tl.abs(q_block))

    # Normalize and quantize to int8, storing results and scaling factor
    tl.store(
        q_int8 + offset_q_int8 + tl.arange(0, 16)[:, None] * BLKQ + tl.arange(0, 16)[None, :]
    )

# Triton kernel for converting key matrices to int8
@triton.jit
def k_kernel_per_block_int8(
    k, k_int8, k_scale, BLKK, BLOCK: tl.constexpr,
):
    # Determine offsets for current program instance
    program_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offset_k = program_id * BLKK * BLKK * 16 + block_id * BLKK * 16
    offset_k_int8 = program_id * BLKK * BLKK + block_id * BLKK
    offset_k_scale = program_id * BLKK * BLKK + block_id * BLKK

    # Load block of key matrix
    k_block = tl.load(
        k + offset_k + tl.arange(0, 16)[:, None] * BLKK + tl.arange(0, 16)[None, :]
    )

    # Compute scaling factor for normalization
    scale = tl.max(tl.abs(k_block))

    # Normalize and quantize to int8, storing results and scaling factor
    tl.store(
        k_int8 + offset_k_int8 + tl.arange(0, 16)[:, None] * BLKK + tl.arange(0, 16)[None, :]
    )

# Function to call the Triton kernels
def per_block_int8(q, k, BLKQ, BLKK):
    # Ensure input tensors are 4D
    if len(q.shape) == 3:
        q = q.unsqueeze(1)
    if len(k.shape) == 3:
        k = k.unsqueeze(1)

    BLOCK = 16

    # Initialize empty int8 and scaling tensors
    q_int8 = torch.empty(
        q.shape[0], q.shape[1], BLKQ, BLKQ, dtype=torch.int8, device=q.device
    )
    k_int8 = torch.empty(
        k.shape[0], k.shape[1], BLKK, BLKK, dtype=torch.int8, device=k.device
    )
    q_scale = torch.empty(
        q.shape[0], q.shape[1], BLKQ, BLKQ, dtype=torch.float16, device=q.device
    )
    k_scale = torch.empty(
        k.shape[0], k.shape[1], BLKK, BLKK, dtype=torch.float16, device=k.device
    )

    # Calculate grid configuration for kernel execution
    grid = lambda meta: (
        q.shape[1] * q.shape[0],
        q.shape[1] * q.shape[0],
    )

    # Launch Triton kernels
    q_kernel_per_block_int8[grid](q, q_int8, q_scale, BLKQ, BLOCK=BLOCK)
    k_kernel_per_block_int8[grid](k, k_int8, k_scale, BLKK, BLOCK=BLOCK)

    return q_int8, q_scale, k_int8, k_scale
