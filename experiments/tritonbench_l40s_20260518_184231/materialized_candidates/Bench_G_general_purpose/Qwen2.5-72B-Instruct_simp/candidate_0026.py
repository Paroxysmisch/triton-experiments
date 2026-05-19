import triton
import triton.language as tl

@triton.jit
def q_kernel_per_block_int8(q, q_int8, q_scale, BLKQ: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLKQ
    q_block = tl.load(q + block_start : block_start + BLKQ)
    max_val = tl.max(q_block, axis=0)
    min_val = tl.min(q_block, axis=0)
    scale = (max_val - min_val) / 255.0
    q_scaled = (q_block - min_val) / scale
    q_int8_block = tl.cast(tl.floor(q_scaled + 0.5), tl.int8)
    tl.store(q_int8 + block_start, q_int8_block)
    tl.store(q_scale + pid, scale)

@triton.jit
def k_kernel_per_block_int8(k, k_int8, k_scale, BLKK: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLKK
    k_block = tl.load(k + block_start : block_start + BLKK)
    max_val = tl.max(k_block, axis=0)
    min_val = tl.min(k_block, axis=0)
    scale = (max_val - min_val) / 255.0
    k_scaled = (k_block - min_val) / scale
    k_int8_block = tl.cast(tl.floor(k_scaled + 0.5), tl.int8)
    tl.store(k_int8 + block_start, k_int8_block)
    tl.store(k_scale + pid, scale)

import torch

def per_block_int8(q, k, BLKQ, BLKK):
    # Ensure the inputs are on the same device
    device = q.device

    # Get the shapes of the input tensors
    q_shape = q.shape
    k_shape = k.shape

    # Calculate the number of blocks for q and k
    num_q_blocks = (q_shape[0] + BLKQ - 1) // BLKQ
    num_k_blocks = (k_shape[0] + BLKK - 1) // BLKK

    # Prepare the output tensors
    q_int8 = torch.empty((q_shape[0], q_shape[1]), dtype=torch.int8, device=device)
    q_scale = torch.empty((num_q_blocks, q_shape[1]), dtype=torch.float32, device=device)
    k_int8 = torch.empty((k_shape[0], k_shape[1]), dtype=torch.int8, device=device)
    k_scale = torch.empty((num_k_blocks, k_shape[1]), dtype=torch.float32, device=device)

    # Launch the kernels
    grid_q = (num_q_blocks, 1, 1)
    grid_k = (num_k_blocks, 1, 1)

    q_kernel_per_block_int8[grid_q](q, q_int8, q_scale, BLKQ)
    k_kernel_per_block_int8[grid_k](k, k_int8, k_scale, BLKK)

    return q_int8, q_scale, k_int8, k_scale
