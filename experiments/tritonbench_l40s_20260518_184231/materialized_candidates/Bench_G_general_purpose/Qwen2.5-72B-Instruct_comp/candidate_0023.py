import triton
import triton.language as tl

# Constants
Q_HEAD_NUM = 16  # Number of query heads
HEAD_DIM = 64    # Dimension of each head

# Rotary Embedding Kernel (without cache)
@triton.jit
def rotary_embedding_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_head_dim_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_head_dim_stride,
    cos_seq_stride, cos_head_dim_stride,
    sin_seq_stride, sin_head_dim_stride,
    batch_size, seq_len, head_num, head_dim,
    BLOCK_SIZE_SEQ: tl.constexpr, BLOCK_SIZE_HEAD_DIM: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(seq_len, BLOCK_SIZE_SEQ)
    num_pid_n = tl.cdiv(head_dim, BLOCK_SIZE_HEAD_DIM)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid %= num_pid_in_batch
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    rm = pid_m * BLOCK_SIZE_SEQ + tl.arange(0, BLOCK_SIZE_SEQ)
    rn = pid_n * BLOCK_SIZE_HEAD_DIM + tl.arange(0, BLOCK_SIZE_HEAD_DIM)
    q_offsets = (
        batch_id * q_batch_stride
        + tl.arange(0, head_num)[:, None, None]
        + rm[None, :, None]
        + rn[None, None, :]
        * q_head_dim_stride
    )
    k_offsets = (
        batch_id * k_batch_stride
        + tl.arange(0, head_num)[:, None, None]
        + rm[None, :, None]
        + rn[None, None, :]
        * k_head_dim_stride
    )
    cos_offsets = (
        rm[None, :]
        + rn[None, :]
        * cos_head_dim_stride
    )
    sin_offsets = (
        rm[None, :]
        + rn[None, :]
        * sin_head_dim_stride
    )
    q_mask = rm < seq_len
    k_mask = rm < seq_len
    q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
    k = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
    cos = tl.load(cos_ptr + cos_offsets, mask=q_mask, other=1.0)
    sin = tl.load(sin_ptr + sin_offsets, mask=q_mask, other=0.0)

    q0 = q[:, 0::2, :]
    q1 = q[:, 1::2, :]
    k0 = k[:, 0::2, :]
    k1 = k[:, 1::2, :]

    out_q0 = q0 * cos - q1 * sin
    out_q1 = q0 * sin + q1 * cos
    out_k0 = k0 * cos - k1 * sin
    out_k1 = k0 * sin + k1 * cos

    out_q = tl.zeros_like(q)
    out_k = tl.zeros_like(k)
    out_q[:, 0::2, :] = out_q0
    out_q[:, 1::2, :] = out_q1
    out_k[:, 0::2, :] = out_k0
    out_k[:, 1::2, :] = out_k1

    tl.store(q_ptr + q_offsets, out_q, mask=q_mask)
    tl.store(k_ptr + k_offsets, out_k, mask=k_mask)

# Fused Rotary Embedding Kernel (with cache)
@triton.jit
def fused_rotary_embedding_kernel_v2(
    q_ptr, k_ptr, k_cache_ptr, cos_ptr, sin_ptr,
    block_tables_ptr, kv_lengths_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_head_dim_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_head_dim_stride,
    k_cache_batch_stride, k_cache_head_stride, k_cache_seq_stride, k_cache_head_dim_stride,
    cos_seq_stride, cos_head_dim_stride,
    sin_seq_stride, sin_head_dim_stride,
    block_tables_batch_stride, block_tables_head_stride, block_tables_seq_stride,
    kv_lengths_batch_stride, kv_lengths_head_stride,
    batch_size, seq_len, head_num, head_dim,
    BLOCK_SIZE_SEQ: tl.constexpr, BLOCK_SIZE_HEAD_DIM: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(seq_len, BLOCK_SIZE_SEQ)
    num_pid_n = tl.cdiv(head_dim, BLOCK_SIZE_HEAD_DIM)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid %= num_pid_in_batch
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    rm = pid_m * BLOCK_SIZE_SEQ + tl.arange(0, BLOCK_SIZE_SEQ)
    rn = pid_n * BLOCK_SIZE_HEAD_DIM + tl.arange(0, BLOCK_SIZE_HEAD_DIM)
    q_offsets = (
        batch_id * q_batch_stride
        + tl.arange(0, head_num)[:, None, None]
        + rm[None, :, None]
        + rn[None, None, :]
        * q_head_dim_stride
    )
    k_offsets = (
        batch_id * k_batch_stride
        + tl.arange(0, head_num)[:, None, None]
        + rm[None, :, None]
        + rn[None, None, :]
        * k_head_dim_stride
    )
    k_cache_offsets = (
        batch_id * k_cache_batch_stride
        + tl.arange(0, head_num)[:, None, None]
        + rm[None, :, None]
        + rn[None, None, :]
        * k_cache_head_dim_stride
    )
    cos_offsets = (
        rm[None, :]
        + rn[None, :]
        * cos_head_dim_stride
    )
    sin_offsets = (
        rm[None, :]
        + rn[None, :]
        * sin_head_dim_stride
    )
    block_tables_offsets = (
        batch_id * block_tables_batch_stride
        + tl.arange(0, head_num)[:, None]
        + rm[None, :]
        * block_tables_seq_stride
    )
    kv_lengths_offsets = (
        batch_id * kv_lengths_batch_stride
        + tl.arange(0, head_num)[:, None]
    )
    q_mask = rm < seq_len
    k_mask = rm < seq_len
    q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
    k = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
    cos = tl.load(cos_ptr + cos_offsets, mask=q_mask, other=1.0)
    sin = tl.load(sin_ptr + sin_offsets, mask=q_mask, other=0.0)
    block_tables = tl.load(block_tables_ptr + block_tables_offsets, mask=q_mask, other=0)
    kv_lengths = tl.load(kv_lengths_ptr + kv_lengths_offsets, mask=q_mask, other=0)

    q0 = q[:, 0::2, :]
    q1 = q[:, 1::2, :]
    k0 = k[:, 0::2, :]
    k1 = k[:, 1::2, :]

    out_q0 = q0 * cos - q1 * sin
    out_q1 = q0 * sin + q1 * cos
    out_k0 = k0 * cos - k1 * sin
    out_k1 = k0 * sin + k1 * cos

    out_q = tl.zeros_like(q)
    out_k = tl.zeros_like(k)
    out_q[:, 0::2, :] = out_q0
    out_q[:, 1::2, :] = out_q1
    out_k[:, 0::2, :] = out_k0
    out_k[:, 1::2, :] = out_k1

    tl.store(q_ptr + q_offsets, out_q, mask=q_mask)
    tl.store(k_ptr + k_offsets, out_k, mask=k_mask)

    # Store to cache
    cache_mask = (rm < kv_lengths)
    tl.store(k_cache_ptr + k_cache_offsets, out_k, mask=cache_mask)

# Wrapper Function
def rotary_embedding(q, k, cos, sin, k_cache=None, block_tables=None, kv_lengths=None):
    batch_size, seq_len, head_num, head_dim = q.shape
    grid = (batch_size * head_num * tl.cdiv(seq_len, 128),)

    if k_cache is None:
        rotary_embedding_kernel[grid](
            q, k, cos, sin,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            batch_size, seq_len, head_num, head_dim,
            BLOCK_SIZE_SEQ=128, BLOCK_SIZE_HEAD_DIM=64
