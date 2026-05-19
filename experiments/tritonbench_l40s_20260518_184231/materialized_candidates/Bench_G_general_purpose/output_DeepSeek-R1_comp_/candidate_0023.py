import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def rotary_embedding_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_head_dim_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_head_dim_stride,
    cos_seq_stride, cos_head_dim_stride,
    sin_seq_stride, sin_head_dim_stride,
    batch_size, num_heads, seq_len, head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_batch_heads_seq = batch_size * num_heads * seq_len
    pid_batch_head_seq = pid // (head_dim // (2 * BLOCK_SIZE))
    pid_block = pid % (head_dim // (2 * BLOCK_SIZE))

    batch_idx = pid_batch_head_seq // (num_heads * seq_len)
    remaining = pid_batch_head_seq % (num_heads * seq_len)
    head_idx = remaining // seq_len
    pos_idx = remaining % seq_len

    cos_pos_offset = pos_idx * cos_seq_stride
    sin_pos_offset = pos_idx * sin_seq_stride

    pair_start = pid_block * BLOCK_SIZE
    offsets = pair_start * 2 + tl.arange(0, BLOCK_SIZE) * 2
    mask = offsets < head_dim
    pair_indices = offsets // 2

    cos_offsets = cos_pos_offset + pair_indices * cos_head_dim_stride
    sin_offsets = sin_pos_offset + pair_indices * sin_head_dim_stride
    cos_vals = tl.load(cos_ptr + cos_offsets, mask=mask)
    sin_vals = tl.load(sin_ptr + sin_offsets, mask=mask)

    q_offset = batch_idx * q_batch_stride + head_idx * q_head_stride + pos_idx * q_seq_stride + offsets
    q0 = tl.load(q_ptr + q_offset, mask=mask)
    q1 = tl.load(q_ptr + q_offset + 1, mask=mask)
    out_q0 = q0 * cos_vals - q1 * sin_vals
    out_q1 = q0 * sin_vals + q1 * cos_vals
    tl.store(q_ptr + q_offset, out_q0, mask=mask)
    tl.store(q_ptr + q_offset + 1, out_q1, mask=mask)

    if k_ptr != 0:
        k_offset = batch_idx * k_batch_stride + head_idx * k_head_stride + pos_idx * k_seq_stride + offsets
        k0 = tl.load(k_ptr + k_offset, mask=mask)
        k1 = tl.load(k_ptr + k_offset + 1, mask=mask)
        out_k0 = k0 * cos_vals - k1 * sin_vals
        out_k1 = k0 * sin_vals + k1 * cos_vals
        tl.store(k_ptr + k_offset, out_k0, mask=mask)
        tl.store(k_ptr + k_offset + 1, out_k1, mask=mask)

@triton.jit
def fused_rotary_embedding_kernel_v2(
    q_ptr, k_cache_ptr, cos_ptr, sin_ptr,
    block_tables_ptr, kv_lengths_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_head_dim_stride,
    k_cache_block_stride, k_cache_head_stride, k_cache_size_stride, k_cache_head_dim_stride,
    cos_seq_stride, cos_head_dim_stride,
    sin_seq_stride, sin_head_dim_stride,
    block_table_batch_stride,
    max_kv_length: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // block_table_batch_stride
    head_idx = pid % block_table_batch_stride

    seq_pos = tl.load(kv_lengths_ptr + batch_idx)
    block_idx_in_table = seq_pos // CACHE_BLOCK_SIZE
    offset_in_block = seq_pos % CACHE_BLOCK_SIZE

    block_table_ptr = block_tables_ptr + batch_idx * block_table_batch_stride
    cache_block_idx = tl.load(block_table_ptr + block_idx_in_table)

    cos_pos = seq_pos * cos_seq_stride
    sin_pos = seq_pos * sin_seq_stride

    range_idx = tl.arange(0, BLOCK_SIZE)
    offsets = range_idx * 2
    mask = offsets < head_dim
    pair_indices = offsets // 2

    cos_vals = tl.load(cos_ptr + cos_pos + pair_indices * cos_head_dim_stride, mask=mask)
    sin_vals = tl.load(sin_ptr + sin_pos + pair_indices * sin_head_dim_stride, mask=mask)

    q_offset = batch_idx * q_batch_stride + head_idx * q_head_stride + seq_pos * q_seq_stride + offsets
    q0 = tl.load(q_ptr + q_offset, mask=mask)
    q1 = tl.load(q_ptr + q_offset + 1, mask=mask)
    out_q0 = q0 * cos_vals - q1 * sin_vals
    out_q1 = q0 * sin_vals + q1 * cos_vals
    tl.store(q_ptr + q_offset, out_q0, mask=mask)
    tl.store(q_ptr + q_offset + 1, out_q1, mask=mask)

    k_cache_ptr += (cache_block_idx * k_cache_block_stride +
                    head_idx * k_cache_head_stride +
                    offset_in_block * k_cache_size_stride +
                    offsets * k_cache_head_dim_stride)
    tl.store(k_cache_ptr, out_q0, mask=mask)
    tl.store(k_cache_ptr + k_cache_head_dim_stride, out_q1, mask=mask)

def rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    kv_lengths: Optional[torch.Tensor] = None,
):
    assert q.shape == k.shape, "q and k must have the same shape"
    batch_size, seq_len, num_heads, head_dim = q.shape
    cos_head_dim = cos.size(-1)
    assert head_dim == 2 * cos_head_dim, "cos/sin dimension mismatch"

    if k_cache is None:
        grid = (batch_size * num_heads * seq_len * (head_dim // (2 * 64)),)
        rotary_embedding_kernel[grid](
            q, k, cos, sin,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            batch_size, num_heads, seq_len, head_dim,
            BLOCK_SIZE=64,
            num_warps=4,
        )
    else:
        assert block_tables is not None and kv_lengths is not None
        grid = (batch_size * num_heads,)
        fused_rotary_embedding_kernel_v2[grid](
            q, k_cache, cos, sin,
            block_tables, kv_lengths,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            block_tables.stride(0),
            max_kv_length=kv_lengths.max().item(),
            head_dim=head_dim,
            BLOCK_SIZE=64,
            CACHE_BLOCK_SIZE=128,
            num_warps=4,
        )
    return q, k if k_cache is None else k_cache
