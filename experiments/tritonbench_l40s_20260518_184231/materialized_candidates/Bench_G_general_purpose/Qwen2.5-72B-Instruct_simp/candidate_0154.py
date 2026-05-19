import triton
import triton.language as tl

@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr, k_cache_ptr, v_cache_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_token_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_token_stride,
    v_batch_stride, v_head_stride, v_seq_stride, v_token_stride,
    cos_batch_stride, cos_head_stride, cos_seq_stride, cos_token_stride,
    sin_batch_stride, sin_head_stride, sin_seq_stride, sin_token_stride,
    k_cache_batch_stride, k_cache_head_stride, k_cache_seq_stride, k_cache_token_stride,
    v_cache_batch_stride, v_cache_head_stride, v_cache_seq_stride, v_cache_token_stride,
    batch_size, num_heads, seq_len, token_dim, rotary_dim, block_m: tl.constexpr, block_n: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(seq_len, block_m)
    num_pid_n = tl.cdiv(token_dim, block_n)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid_in_batch = pid % num_pid_in_batch
    nm = pid_in_batch
    pid_m = nm // num_pid_n
    pid_n = nm % num_pid_n
    block_start_m = pid_m * block_m
    block_start_n = pid_n * block_n

    offsets_m = block_start_m + tl.arange(0, block_m)
    offsets_n = block_start_n + tl.arange(0, block_n)
    offsets_q = batch_id * q_batch_stride + tl.arange(0, num_heads) * q_head_stride + offsets_m[:, None] * q_seq_stride + offsets_n[None, :] * q_token_stride
    offsets_k = batch_id * k_batch_stride + tl.arange(0, num_heads) * k_head_stride + offsets_m[:, None] * k_seq_stride + offsets_n[None, :] * k_token_stride
    offsets_v = batch_id * v_batch_stride + tl.arange(0, num_heads) * v_head_stride + offsets_m[:, None] * v_seq_stride + offsets_n[None, :] * v_token_stride
    offsets_cos = batch_id * cos_batch_stride + tl.arange(0, num_heads) * cos_head_stride + offsets_m[:, None] * cos_seq_stride + offsets_n[None, :] * cos_token_stride
    offsets_sin = batch_id * sin_batch_stride + tl.arange(0, num_heads) * sin_head_stride + offsets_m[:, None] * sin_seq_stride + offsets_n[None, :] * sin_token_stride
    offsets_k_cache = batch_id * k_cache_batch_stride + tl.arange(0, num_heads) * k_cache_head_stride + offsets_m[:, None] * k_cache_seq_stride + offsets_n[None, :] * k_cache_token_stride
    offsets_v_cache = batch_id * v_cache_batch_stride + tl.arange(0, num_heads) * v_cache_head_stride + offsets_m[:, None] * v_cache_seq_stride + offsets_n[None, :] * v_cache_token_stride

    q = tl.load(q_ptr + offsets_q, mask=offsets_m[:, None] < seq_len, other=0.0)
    k = tl.load(k_ptr + offsets_k, mask=offsets_m[:, None] < seq_len, other=0.0)
    v = tl.load(v_ptr + offsets_v, mask=offsets_m[:, None] < seq_len, other=0.0)
    cos = tl.load(cos_ptr + offsets_cos, mask=offsets_m[:, None] < seq_len, other=1.0)
    sin = tl.load(sin_ptr + offsets_sin, mask=offsets_m[:, None] < seq_len, other=0.0)

    q_rot = q * cos - tl.flip(q, 1) * sin
    k_rot = k * cos - tl.flip(k, 1) * sin

    tl.store(q_ptr + offsets_q, q_rot, mask=offsets_m[:, None] < seq_len)
    tl.store(k_ptr + offsets_k, k_rot, mask=offsets_m[:, None] < seq_len)

    tl.store(k_cache_ptr + offsets_k_cache, k_rot, mask=offsets_m[:, None] < seq_len)
    tl.store(v_cache_ptr + offsets_v_cache, v, mask=offsets_m[:, None] < seq_len)

import torch

def decoding_fused_rotary_embedding(
    q, k, v, cos, sin, k_cache, v_cache,
    batch_size, num_heads, seq_len, token_dim, rotary_dim,
    block_m=128, block_n=64
):
    assert q.shape == (batch_size, num_heads, seq_len, token_dim)
    assert k.shape == (batch_size, num_heads, seq_len, token_dim)
    assert v.shape == (batch_size, num_heads, seq_len, token_dim)
    assert cos.shape == (batch_size, num_heads, seq_len, token_dim)
    assert sin.shape == (batch_size, num_heads, seq_len, token_dim)
    assert k_cache.shape == (batch_size, num_heads, seq_len, token_dim)
    assert v_cache.shape == (batch_size, num_heads, seq_len, token_dim)

    q_batch_stride = q.stride(0)
    q_head_stride = q.stride(1)
    q_seq_stride = q.stride(2)
    q_token_stride = q.stride(3)

    k_batch_stride = k.stride(0)
    k_head_stride = k.stride(1)
    k_seq_stride = k.stride(2)
    k_token_stride = k.stride(3)

    v_batch_stride = v.stride(0)
    v_head_stride = v.stride(1)
    v_seq_stride = v.stride(2)
    v_token_stride = v.stride(3)

    cos_batch_stride = cos.stride(0)
    cos_head_stride = cos.stride(1)
    cos_seq_stride = cos.stride(2)
    cos_token_stride = cos.stride(3)

    sin_batch_stride = sin.stride(0)
    sin_head_stride = sin.stride(1)
    sin_seq_stride = sin.stride(2)
    sin_token_stride = sin.stride(3)

    k_cache_batch_stride = k_cache.stride(0)
    k_cache_head_stride = k_cache.stride(1)
    k_cache_seq_stride = k_cache.stride(2)
    k_cache_token_stride = k_cache.stride(3)

    v_cache_batch_stride = v_cache.stride(0)
    v_cache_head_stride = v_cache.stride(1)
    v_cache_seq_stride = v_cache.stride(2)
    v_cache_token_stride = v_cache.stride(3)

    grid = (batch_size * num_heads * tl.cdiv(seq_len, block_m) * tl.cdiv(token_dim, block_n),)

    decoding_fused_rotary_embedding_kernel[grid](
        q, k, v, cos, sin, k_cache, v_cache,
        q_batch_stride, q_head_stride, q_seq_stride, q_token_stride,
        k_batch_stride, k_head_stride, k_seq_stride, k_token_stride,
        v_batch_stride, v_head_stride, v_seq_stride, v_token_stride,
        cos_batch_stride, cos_head_stride, cos_seq_stride, cos_token_stride,
        sin_batch_stride, sin_head_stride, sin_seq_stride, sin_token_stride,
        k_cache_batch_stride, k_cache_head_stride, k_cache_seq_stride, k_cache_token_stride,
        v_cache_batch_stride, v_cache_head_stride, v_cache_seq_stride, v_cache_token_stride,
        batch_size, num_heads, seq_len, token_dim, rotary_dim, block_m, block_n
    )
