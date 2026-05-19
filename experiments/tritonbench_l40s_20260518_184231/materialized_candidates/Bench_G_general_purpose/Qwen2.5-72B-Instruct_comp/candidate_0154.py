import triton
import triton.language as tl

@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q_ptr, k_ptr, v_ptr, k_cache_ptr, v_cache_ptr,
    q_head_num, q_total_tokens, head_dim, rotary_dim,
    q_stride0, q_stride1, q_stride2, q_stride3,
    k_stride0, k_stride1, k_stride2, k_stride3,
    v_stride0, v_stride1, v_stride2, v_stride3,
    k_cache_stride0, k_cache_stride1, k_cache_stride2, k_cache_stride3,
    v_cache_stride0, v_cache_stride1, v_cache_stride2, v_cache_stride3,
    block_tables, kv_lengths, kv_group_num,
    use_new_kcache_layout: tl.constexpr,
    rotary_base: tl.constexpr,
    rotary_scale: tl.constexpr,
    BLOCK_SIZE_HEAD: tl.constexpr,
    BLOCK_SIZE_TOKEN: tl.constexpr,
    WARP_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    head_id = pid // q_total_tokens
    token_id = pid % q_total_tokens

    head_start = head_id * BLOCK_SIZE_HEAD
    token_start = token_id * BLOCK_SIZE_TOKEN

    rotary_half_dim = rotary_dim // 2
    rotary_scale_inv = 1.0 / rotary_scale

    for head in range(head_start, head_start + BLOCK_SIZE_HEAD):
        if head >= q_head_num:
            break

        for token in range(token_start, token_start + BLOCK_SIZE_TOKEN):
            if token >= q_total_tokens:
                break

            q_offset = q_stride0 * head + q_stride1 * token
            k_offset = k_stride0 * head + k_stride1 * token
            v_offset = v_stride0 * head + v_stride1 * token

            q_ptr_head = q_ptr + q_offset
            k_ptr_head = k_ptr + k_offset
            v_ptr_head = v_ptr + v_offset

            q = tl.load(q_ptr_head + q_stride2 * tl.arange(0, head_dim))
            k = tl.load(k_ptr_head + k_stride2 * tl.arange(0, head_dim))
            v = tl.load(v_ptr_head + v_stride2 * tl.arange(0, head_dim))

            # Compute rotary embeddings
            theta = rotary_base ** (tl.arange(0, rotary_half_dim) / rotary_half_dim)
            freqs = 1.0 / (theta * rotary_scale_inv)
            t = token
            freqs = freqs * t

            cos = tl.cos(freqs)
            sin = tl.sin(freqs)

            q_rot = q[rotary_half_dim:] * cos + q[:rotary_half_dim] * sin
            q_rot = tl.cat(q[:rotary_half_dim] * cos - q[rotary_half_dim:] * sin, q_rot)

            k_rot = k[rotary_half_dim:] * cos + k[:rotary_half_dim] * sin
            k_rot = tl.cat(k[:rotary_half_dim] * cos - k[rotary_half_dim:] * sin, k_rot)

            # Update q with rotary embeddings
            tl.store(q_ptr_head + q_stride2 * tl.arange(0, head_dim), q_rot)

            # Update k and v caches
            if use_new_kcache_layout:
                k_cache_offset = k_cache_stride0 * (head // kv_group_num) + k_cache_stride1 * (head % kv_group_num) + k_cache_stride2 * token
                v_cache_offset = v_cache_stride0 * (head // kv_group_num) + v_cache_stride1 * (head % kv_group_num) + v_cache_stride2 * token
            else:
                k_cache_offset = k_cache_stride0 * head + k_cache_stride1 * token
                v_cache_offset = v_cache_stride0 * head + v_cache_stride1 * token

            k_cache_ptr_head = k_cache_ptr + k_cache_offset
            v_cache_ptr_head = v_cache_ptr + v_cache_offset

            tl.store(k_cache_ptr_head + k_cache_stride2 * tl.arange(0, head_dim), k_rot)
            tl.store(v_cache_ptr_head + v_cache_stride2 * tl.arange(0, head_dim), v)

import torch
import triton
import triton.language as tl

def decoding_fused_rotary_embedding(
    q, k, v, k_cache, v_cache,
    block_tables, kv_lengths, kv_group_num,
    use_new_kcache_layout, rotary_base, rotary_scale
):
    assert q.dim() == 4, "q must be a 4D tensor"
    assert k.dim() == 4, "k must be a 4D tensor"
    assert v.dim() == 4, "v must be a 4D tensor"
    assert k_cache.dim() == 4, "k_cache must be a 4D tensor"
    assert v_cache.dim() == 4, "v_cache must be a 4D tensor"

    q_head_num, q_total_tokens, head_dim = q.shape[0], q.shape[1], q.shape[-1]
    rotary_dim = head_dim // 2

    q_stride0, q_stride1, q_stride2, q_stride3 = q.stride(0), q.stride(1), q.stride(2), q.stride(3)
    k_stride0, k_stride1, k_stride2, k_stride3 = k.stride(0), k.stride(1), k.stride(2), k.stride(3)
    v_stride0, v_stride1, v_stride2, v_stride3 = v.stride(0), v.stride(1), v.stride(2), v.stride(3)
    k_cache_stride0, k_cache_stride1, k_cache_stride2, k_cache_stride3 = k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3)
    v_cache_stride0, v_cache_stride1, v_cache_stride2, v_cache_stride3 = v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3)

    grid = (q_head_num * q_total_tokens,)

    BLOCK_SIZE_HEAD = 1
    BLOCK_SIZE_TOKEN = 1
    WARP_SIZE = 32

    decoding_fused_rotary_embedding_kernel[grid](
        q, k, v, k_cache, v_cache,
        q_head_num, q_total_tokens, head_dim, rotary_dim,
        q_stride0, q_stride1, q_stride2, q_stride3,
        k_stride0, k_stride1, k_stride2, k_stride3,
        v_stride0, v_stride1, v_stride2, v_stride3,
        k_cache_stride0, k_cache_stride1, k_cache_stride2, k_cache_stride3,
        v_cache_stride0, v_cache_stride1, v_cache_stride2, v_cache_stride3,
        block_tables, kv_lengths, kv_group_num,
        use_new_kcache_layout,
        rotary_base, rotary_scale,
        BLOCK_SIZE_HEAD, BLOCK_SIZE_TOKEN, WARP_SIZE
    )

    return q, k_cache, v_cache
