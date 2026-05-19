import torch
import triton
import triton.language as tl

# Triton kernel for converting query matrix to int8
@triton.jit
def q_kernel_per_block_int8(
    start_idx,
    q,
    q_int8,
    q_scale,
    D,
    L,
    BLKQ: tl.constexpr,
):
    off_d = tl.arange(0, BLKQ)
    off_l = tl.arange(0, BLKQ)
    off_q = start_idx + off_d[:, None] * D + off_l[None, :] * L
    q_ptrs = q + off_q
    mask = (off_d[:, None] < D) & (off_l[None, :] < L)
    q_block = tl.load(q_ptrs, mask=mask).to(tl.float32)
    q_block = tl.where(mask, q_block, 0)
    D_block = tl.max(tl.abs(q_block), axis=1)[:, None]
    scale_block = D_block / 127.0
    q_int8_block = (q_block / scale_block).to(tl.int8)
    q_int8_block = tl.where(mask, q_int8_block, 0)
    q_scale_block = scale_block
    off_q = start_idx + off_d[:, None] * D + off_l[None, :] * L
    q_int8_ptrs = q_int8 + off_q
    q_scale_ptrs = q_scale + off_d
    tl.store(q_int8_ptrs, q_int8_block, mask=mask)
    tl.store(q_scale_ptrs, q_scale_block, mask=off_d < D)

# Triton kernel for converting key matrix to int8
@triton.jit
def k_kernel_per_block_int8(
    start_idx,
    k,
    k_int8,
    k_scale,
    D,
    L,
    BLKK: tl.constexpr,
):
    off_d = tl.arange(0, BLKK)
    off_l = tl.arange(0, BLKK)
    off_k = start_idx + off_d[:, None] * D + off_l[None, :] * L
    k_ptrs = k + off_k
    mask = (off_d[:, None] < D) & (off_l[None, :] < L)
    k_block = tl.load(k_ptrs, mask=mask).to(tl.float32)
    k_block = tl.where(mask, k_block, 0)
    D_block = tl.max(tl.abs(k_block), axis=1)[:, None]
    scale_block = D_block / 127.0
    k_int8_block = (k_block / scale_block).to(tl.int8)
    k_int8_block = tl.where(mask, k_int8_block, 0)
    k_scale_block = scale_block
    off_k = start_idx + off_d[:, None] * D + off_l[None, :] * L
    k_int8_ptrs = k_int8 + off_k
    k_scale_ptrs = k_scale + off_d
    tl.store(k_int8_ptrs, k_int8_block, mask=mask)
    tl.store(k_scale_ptrs, k_scale_block, mask=off_d < D)

# Wrapper function to call Triton kernels
def per_block_int8(q, k, BLKQ, BLKK):
    B, H, D, L = q.shape
    q = q.reshape((B * H, D, L))
    k = k.reshape((B * H, D, L))
    q_int8 = torch.empty((B * H, BLKQ, BLKQ), dtype=torch.int8, device=q.device)
    q_scale = torch.empty((B * H, D), dtype=torch.float16, device=q.device)
    k_int8 = torch.empty((B * H, BLKK, BLKK), dtype=torch.int8, device=q.device)
    k_scale = torch.empty((B * H, D), dtype=torch.float16, device=q.device)
    grid = (triton.cdiv(D, BLKQ), B * H)
    q_kernel_per_block_int8[grid](
        0,
        q,
        q_int8,
        q_scale,
        D,
        L,
        BLKQ=BLKQ,
    )
    grid = (triton.cdiv(D, BLKK), B * H)
    k_kernel_per_block_int8[grid](
        0,
        k,
        k_int8,
        k_scale,
        D,
        L,
        BLKK=BLKK,
    )
    q_int8 = q_int8.reshape((B, H, q_int8.shape[1], q_int8.shape[2]))
    k_int8 = k_int8.reshape((B, H, k_int8.shape[1], k_int8.shape[2]))
    return q_int8, k_int8, q_scale, k_scale
