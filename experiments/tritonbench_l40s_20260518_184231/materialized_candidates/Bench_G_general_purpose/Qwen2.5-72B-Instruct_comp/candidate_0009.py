import math

def calculate_settings(n, max_fused_size=1024):
    BLOCK_SIZE = 1
    while BLOCK_SIZE < n:
        BLOCK_SIZE *= 2
    if BLOCK_SIZE > max_fused_size:
        raise RuntimeError(f"Block size {BLOCK_SIZE} exceeds maximum allowed size {max_fused_size}")
    num_warps = 4 if BLOCK_SIZE <= 512 else 8
    return BLOCK_SIZE, num_warps

import triton
import triton.language as tl

@triton.jit
def _rope_embedding(
    Q, Q_row_stride, cos, cos_row_stride, sin, sin_row_stride,
    seqlen, head_dim, n_heads, BACKWARD_PASS,
    BLOCK_SIZE: tl.constexpr, ROPE_GROUP_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    group_id = pid // (seqlen * n_heads // ROPE_GROUP_SIZE)
    head_id = (pid % (seqlen * n_heads // ROPE_GROUP_SIZE)) // seqlen
    seq_id = (pid % (seqlen * n_heads // ROPE_GROUP_SIZE)) % seqlen

    Q_offset = (group_id * ROPE_GROUP_SIZE + head_id) * head_dim + seq_id * Q_row_stride
    cos_offset = seq_id * cos_row_stride
    sin_offset = seq_id * sin_row_stride

    q = tl.load(Q + Q_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)
    cos_val = tl.load(cos + cos_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=1.0)
    sin_val = tl.load(sin + sin_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)

    q1 = q[:head_dim // 2]
    q2 = q[head_dim // 2:]

    if not BACKWARD_PASS:
        q1 = q1 * cos_val[:head_dim // 2] - q2 * sin_val[:head_dim // 2]
        q2 = q1 * sin_val[:head_dim // 2] + q2 * cos_val[:head_dim // 2]
    else:
        q1 = q1 * cos_val[:head_dim // 2] + q2 * sin_val[:head_dim // 2]
        q2 = q2 * cos_val[:head_dim // 2] - q1 * sin_val[:head_dim // 2]

    q = tl.cat(q1, q2)
    tl.store(Q + Q_offset + tl.arange(0, BLOCK_SIZE), q, mask=tl.arange(0, BLOCK_SIZE) < head_dim)

import torch

def _rope_embedding_forward_impl(Q, cos, sin, ROPE_GROUP_SIZE=1):
    seqlen, head_dim, n_heads = Q.shape[0], Q.shape[1], Q.shape[2] // head_dim
    BLOCK_SIZE, num_warps = calculate_settings(head_dim)
    n_groups = n_heads // ROPE_GROUP_SIZE

    Q = Q.view(seqlen, n_heads, head_dim).transpose(0, 1).contiguous()
    cos = cos.view(seqlen, head_dim).contiguous()
    sin = sin.view(seqlen, head_dim).contiguous()

    grid = (n_groups * seqlen, )
    _rope_embedding[grid](
        Q, head_dim, cos, head_dim, sin, head_dim,
        seqlen, head_dim, n_heads, False,
        BLOCK_SIZE, ROPE_GROUP_SIZE,
        num_warps=num_warps
    )

    return Q.view(n_heads, seqlen, head_dim).transpose(0, 1).contiguous()

def _rope_embedding_backward_impl(dY, cos, sin, n_groups, BLOCK_SIZE, num_warps, ROPE_GROUP_SIZE=1):
    seqlen, head_dim, n_heads = dY.shape[0], dY.shape[1], dY.shape[2] // head_dim

    dY = dY.view(seqlen, n_heads, head_dim).transpose(0, 1).contiguous()
    cos = cos.view(seqlen, head_dim).contiguous()
    sin = sin.view(seqlen, head_dim).contiguous()

    grid = (n_groups * seqlen, )
    _rope_embedding[grid](
        dY, head_dim, cos, head_dim, sin, head_dim,
        seqlen, head_dim, n_heads, True,
        BLOCK_SIZE, ROPE_GROUP_SIZE,
        num_warps=num_warps
    )

    return dY.view(n_heads, seqlen, head_dim).transpose(0, 1).contiguous()
