import triton
import triton.language as tl

# Constants
Q_HEAD_NUM = 16  # Number of heads
HEAD_DIM = 64    # Dimension of each head

@triton.jit
def rotary_embedding_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr, q_head_num, head_dim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    head_id = pid // (head_dim // BLOCK_SIZE)
    block_id = pid % (head_dim // BLOCK_SIZE)

    q_offset = (head_id * head_dim + block_id * BLOCK_SIZE) * tl.int8
    k_offset = (head_id * head_dim + block_id * BLOCK_SIZE) * tl.int8
    cos_offset = (block_id * BLOCK_SIZE) * tl.int8
    sin_offset = (block_id * BLOCK_SIZE) * tl.int8

    q = tl.load(q_ptr + q_offset, mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)
    k = tl.load(k_ptr + k_offset, mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)
    cos = tl.load(cos_ptr + cos_offset, mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=1.0)
    sin = tl.load(sin_ptr + sin_offset, mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)

    q_rotated = q * cos - k * sin
    k_rotated = q * sin + k * cos

    tl.store(q_ptr + q_offset, q_rotated, mask=tl.arange(0, BLOCK_SIZE) < head_dim)
    tl.store(k_ptr + k_offset, k_rotated, mask=tl.arange(0, BLOCK_SIZE) < head_dim)

@triton.jit
def fused_rotary_embedding_kernel_v2(
    q_ptr, k_ptr, k_cache_ptr, cos_ptr, sin_ptr, q_head_num, head_dim,
    block_table_ptr, context_length_ptr, block_size: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    head_id = pid // (head_dim // BLOCK_SIZE)
    block_id = pid % (head_dim // BLOCK_SIZE)

    q_offset = (head_id * head_dim + block_id * BLOCK_SIZE) * tl.int8
    k_offset = (head_id * head_dim + block_id * BLOCK_SIZE) * tl.int8
    cos_offset = (block_id * BLOCK_SIZE) * tl.int8
    sin_offset = (block_id * BLOCK_SIZE) * tl.int8

    q = tl.load(q_ptr + q_offset, mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)
    k = tl.load(k_ptr + k_offset, mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)
    cos = tl.load(cos_ptr + cos_offset, mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=1.0)
    sin = tl.load(sin_ptr + sin_offset, mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)

    q_rotated = q * cos - k * sin
    k_rotated = q * sin + k * cos

    tl.store(q_ptr + q_offset, q_rotated, mask=tl.arange(0, BLOCK_SIZE) < head_dim)

    block_idx = tl.load(block_table_ptr + head_id * block_size + block_id)
    context_length = tl.load(context_length_ptr + head_id)
    k_cache_offset = (head_id * head_dim + block_idx * BLOCK_SIZE) * tl.int8

    tl.store(k_cache_ptr + k_cache_offset, k_rotated, mask=tl.arange(0, BLOCK_SIZE) < context_length)

### Python Wrapper

import torch

def rotary_embedding(q, k, cos, sin, k_cache=None, block_table=None, context_length=None, block_size=None):
    q_head_num, head_dim = q.shape
    assert q_head_num == Q_HEAD_NUM
    assert head_dim == HEAD_DIM

    if k_cache is None:
        grid = (Q_HEAD_NUM * (HEAD_DIM // 32),)
        rotary_embedding_kernel[grid](
            q, k, cos, sin, Q_HEAD_NUM, HEAD_DIM, BLOCK_SIZE=32
        )
    else:
        assert block_table is not None
        assert context_length is not None
        assert block_size is not None
        grid = (Q_HEAD_NUM * (HEAD_DIM // 32),)
        fused_rotary_embedding_kernel_v2[grid](
            q, k, k_cache, cos, sin, Q_HEAD_NUM, HEAD_DIM, block_table, context_length, block_size, BLOCK_SIZE=32
        )

    return q, k
