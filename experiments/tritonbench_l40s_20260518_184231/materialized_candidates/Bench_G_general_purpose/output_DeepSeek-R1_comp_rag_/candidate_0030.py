import torch
import triton
import triton.language as tl

@triton.jit
def q_kernel_per_block_int8(
    q_ptr,
    q_int8_ptr,
    scale_ptr,
    M,  # Total rows in the query matrix
    D,  # Total features per row
    BLKQ: tl.constexpr,  # Block size for rows in query
    D_MODEL: tl.constexpr,  # Feature dimension size, must be <= D
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLKQ
    rows = row_start + tl.arange(0, BLKQ)
    row_mask = rows < M

    col_idx = tl.arange(0, D_MODEL)
    col_mask = col_idx < D

    block_mask = row_mask[:, None] & col_mask[None, :]
    block = tl.load(q_ptr + rows[:, None] * D + col_idx[None, :], mask=block_mask, other=0.0)
    abs_block = tl.abs(block)
    max_val = tl.max(abs_block)
    max_val_safe = tl.where(max_val == 0.0, 1.0, max_val)
    scale = max_val_safe / 127.0

    quantized_block = tl.round(block / scale).to(tl.int8)
    tl.store(q_int8_ptr + rows[:, None] * D + col_idx[None, :], quantized_block, mask=block_mask)
    tl.store(scale_ptr + pid, scale)

@triton.jit
def k_kernel_per_block_int8(
    k_ptr,
    k_int8_ptr,
    scale_ptr,
    N,  # Total rows in the key matrix
    D,  # Total features per row
    BLKK: tl.constexpr,  # Block size for rows in key
    D_MODEL: tl.constexpr,  # Feature dimension size, must be <= D
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLKK
    rows = row_start + tl.arange(0, BLKK)
    row_mask = rows < N

    col_idx = tl.arange(0, D_MODEL)
    col_mask = col_idx < D

    block_mask = row_mask[:, None] & col_mask[None, :]
    block = tl.load(k_ptr + rows[:, None] * D + col_idx[None, :], mask=block_mask, other=0.0)
    abs_block = tl.abs(block)
    max_val = tl.max(abs_block)
    max_val_safe = tl.where(max_val == 0.0, 1.0, max_val)
    scale = max_val_safe / 127.0

    quantized_block = tl.round(block / scale).to(tl.int8)
    tl.store(k_int8_ptr + rows[:, None] * D + col_idx[None, :], quantized_block, mask=block_mask)
    tl.store(scale_ptr + pid, scale)

def per_block_int8(q, k, BLKQ, BLKK):
    # Reshape q to 2D if necessary (handles 3D/4D inputs)
    q_2d = q.view(-1, q.size(-1)) if q.dim() > 2 else q
    M, D_q = q_2d.shape
    # Reshape k to 2D if necessary
    k_2d = k.view(-1, k.size(-1)) if k.dim() > 2 else k
    N, D_k = k_2d.shape

    # Allocate output tensors
    q_int8 = torch.empty_like(q_2d, dtype=torch.int8)
    q_scale = torch.empty((triton.cdiv(M, BLKQ),), device=q.device, dtype=torch.float32)
    k_int8 = torch.empty_like(k_2d, dtype=torch.int8)
    k_scale = torch.empty((triton.cdiv(N, BLKK),), device=k.device, dtype=torch.float32)

    # Grid configurations
    grid_q = (triton.cdiv(M, BLKQ),)
    grid_k = (triton.cdiv(N, BLKK),)

    # Launch query kernel
    q_kernel_per_block_int8[grid_q](
        q_2d, q_int8, q_scale,
        M, D_q,
        BLKQ=BLKQ,
        D_MODEL=D_q,
    )
    # Launch key kernel
    k_kernel_per_block_int8[grid_k](
        k_2d, k_int8, k_scale,
        N, D_k,
        BLKK=BLKK,
        D_MODEL=D_k,
    )

    # Reshape outputs to match input dimensions
    if q.dim() > 2:
        q_int8 = q_int8.view(*q.shape[:-1], -1)
        q_scale = q_scale.view(*q.shape[:-1], -1)[..., :1].expand(*q.shape[:-1], D_q)
    if k.dim() > 2:
        k_int8 = k_int8.view(*k.shape[:-1], -1)
        k_scale = k_scale.view(*k.shape[:-1], -1)[..., :1].expand(*k.shape[:-1], D_k)

    return q_int8, k_int8, q_scale, k_scale
