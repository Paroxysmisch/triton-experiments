import triton
import triton.language as tl
import torch

@triton.jit
def q_kernel_per_block_int8(Q, Q_INT8, Q_SCALE, BLKQ, stride_q, stride_q_int8, stride_q_scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE

    q_block = tl.load(Q + offset * stride_q, mask=offset + tl.arange(0, BLOCK_SIZE) < BLKQ, other=0.0)
    max_val = tl.max(tl.abs(q_block), axis=0)
    scale = 127.0 / max_val
    q_int8_block = tl.cast(q_block * scale, tl.int8)

    tl.store(Q_INT8 + offset * stride_q_int8, q_int8_block, mask=offset + tl.arange(0, BLOCK_SIZE) < BLKQ)
    tl.store(Q_SCALE + pid * stride_q_scale, scale)

@triton.jit
def k_kernel_per_block_int8(K, K_INT8, K_SCALE, BLKK, stride_k, stride_k_int8, stride_k_scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE

    k_block = tl.load(K + offset * stride_k, mask=offset + tl.arange(0, BLOCK_SIZE) < BLKK, other=0.0)
    max_val = tl.max(tl.abs(k_block), axis=0)
    scale = 127.0 / max_val
    k_int8_block = tl.cast(k_block * scale, tl.int8)

    tl.store(K_INT8 + offset * stride_k_int8, k_int8_block, mask=offset + tl.arange(0, BLOCK_SIZE) < BLKK)
    tl.store(K_SCALE + pid * stride_k_scale, scale)

def per_block_int8(q, k, BLKQ, BLKK):
    BLOCK_SIZE = 128  # This can be adjusted based on your GPU's capability

    # Prepare output tensors
    q_int8 = torch.empty_like(q, dtype=torch.int8)
    q_scale = torch.empty((q.size(0) // BLKQ,), dtype=torch.float32)
    k_int8 = torch.empty_like(k, dtype=torch.int8)
    k_scale = torch.empty((k.size(0) // BLKK,), dtype=torch.float32)

    # Launch Triton kernels
    grid_q = (q.size(0) // BLOCK_SIZE,)
    grid_k = (k.size(0) // BLOCK_SIZE,)

    q_kernel_per_block_int8[grid_q](
        q, q_int8, q_scale, BLKQ,
        q.stride(0), q_int8.stride(0), q_scale.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )

    k_kernel_per_block_int8[grid_k](
        k, k_int8, k_scale, BLKK,
        k.stride(0), k_int8.stride(0), k_scale.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )

    return q_int8, q_scale, k_int8, k_scale
