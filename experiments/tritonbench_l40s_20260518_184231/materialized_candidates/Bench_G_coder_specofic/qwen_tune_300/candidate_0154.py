import torch
import triton
import triton.language as tl

@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q, k, v, q_cache, k_cache, v_cache, 
    seq_len, block_tables, kv_lengths,
    rotary_embedding_dim,
    head_dim_square_half: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    LAYOUT_V2: tl.constexpr,
    UPDATE_KVCACHE: tl.constexpr,
    SELECT_KVCACHE: tl.constexpr,
):
    q_head_index = tl.program_id(axis=0)
    q_token_index = tl.program_id(axis=1)

    q_token_offset = q_token_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    q_token_mask = q_token_offset < seq_len

    q_head_offset = q_head_index * rotary_embedding_dim
    q_head_mask = q_head_offset + tl.arange(0, HEAD_DIM) < rotary_embedding_dim

    q_offset = q_token_offset[:, None] * rotary_embedding_dim + q_head_offset + tl.arange(0, HEAD_DIM)
    q_mask = q_token_mask[:, None] & q_head_mask[None, :]

    q_value = tl.load(q + q_offset, mask=q_mask).to(tl.float32)

    k_start_index = tl.cdiv(q_token_index * KV_GROUP_NUM, BLOCK_SIZE)
    v_start_index = tl.cdiv(q_token_index * KV_GROUP_NUM, BLOCK_SIZE)

    for i in range(0, KV_GROUP_NUM):
        k_token_index = k_start_index + i
        v_token_index = v_start_index + i

        k_token_offset = k_token_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        k_token_mask = k_token_offset < kv_lengths
        k_head_offset = tl.arange(0, HEAD_DIM)
        k_head_mask = True

        k_offset = k_token_offset[:, None] * rotary_embedding_dim + q_head_offset + k_head_offset[None, :]
        k_mask = k_token_mask[:, None] & k_head_mask[None, :]

        k_value = tl.load(k + k_offset, mask=k_mask).to(tl.float32)

        v_token_offset = v_token_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        v_token_mask = v_token_offset < kv_lengths
        v_head_offset = tl.arange(0, HEAD_DIM)
        v_head_mask = True

        v_offset = v_token_offset[:, None] * rotary_embedding_dim + q_head_offset + v_head_offset[None, :]
        v_mask = v_token_mask[:, None] & v_head_mask[None, :]

        v_value = tl.load(v + v_offset, mask=v_mask).to(tl.float32)

        if UPDATE_KVCACHE:
            k_cache_offset = k_token_offset[:, None] * HEAD_DIM + q_head_offset + k_head_offset[None, :]
            k_cache_mask = k_token_mask[:, None] & k_head_mask[None, :]
            tl.store(k_cache + k_cache_offset, k_value, mask=k_cache_mask)

            v_cache_offset = v_token_offset[:, None] * HEAD_DIM + q_head_offset + v_head_offset[None, :]
            v_cache_mask = v_token_mask[:, None] & v_head_mask[None, :]
            tl.store(v_cache + v_cache_offset, v_value, mask=v_cache_mask)

        if SELECT_KVCACHE:
            k_cache_offset = k_token_offset[:, None] * HEAD_DIM + q_head_offset + k_head_offset[None, :]
            k_cache_mask = k_token_mask[:, None] & k_head_mask[None, :]
            k_value = tl.load(k_cache + k_cache_offset, mask=k_cache_mask)

            v_cache_offset = v_token_offset[:, None] * HEAD_DIM + q_head_offset + v_head_offset[None, :]
            v_cache_mask = v_token_mask[:, None] & v_head_mask[None, :]
            v_value = tl.load(v_cache + v_cache_offset, mask=v_cache_mask)

        q_cos_offset = q_offset + HEAD_DIM // 2
        q_cos_value = tl.load(q + q_cos_offset, mask=q_mask).to(tl.float32)

        q_sin_offset = q_offset
        q_sin_value = tl.load(q + q_sin_offset, mask=q_mask).to(tl.float32)

        q_sin_value = q_sin_value * tl.math.cosf(k_value * head_dim_square_half)
        q_cos_value = q_cos_value * tl.math.cosf(v_value * head_dim_square_half)

        tl.store(q + q_sin_offset, q_sin_value.to(q.dtype.element_ty), mask=q_mask)
        tl.store(q + q_cos_offset, q_cos_value.to(q.dtype.element_ty), mask=q_mask)


def decoding_fused_rotary_embedding(q, k, v, q_cache, k_cache, v_cache, block_tables, kv_lengths, rotary_embedding_dim, use_new_kcache_layout=False):
    seq_len = q.shape[0]
    head_dim = q.shape[-1] // 2
    head_dim_square = head_dim * head_dim
    head_dim_square_half = head_dim_square // 2
    q_total_tokens = q.shape[0]
    q_head_num = q.shape[1]

    if use_new_kcache_layout:
        kv_lengths = kv_lengths // BLOCK_SIZE * BLOCK_SIZE
        kv_head_num = k_cache.shape[1]
        k_cache = k_cache.view(q.shape[0], kv_head_num * BLOCK_SIZE, -1)
        v_cache = v_cache.view(q.shape[0], kv_head_num * BLOCK_SIZE, -1)
    else:
        kv_head_num = k_cache.shape[2]
        kv_lengths = v_cache.shape[1] * BLOCK_SIZE

    BLOCK_SIZE = max_power_of_2_divisor(min(q.shape[0], k.shape[0], v.shape[0], q_cache.shape[0], k_cache.shape[0], v_cache.shape[0]))
    grid = (q_head_num, q_total_tokens)
    num_warps = get_num_warps(head_dim)

    decoding_fused_rotary_embedding_kernel[grid](
        q, k, v, q_cache, k_cache, v_cache,
        seq_len, block_tables, kv_lengths,
        rotary_embedding_dim,
        head_dim_square_half,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        KV_GROUP_NUM=BLOCK_SIZE // kv_lengths,
        LAYOUT_V2=use_new_kcache_layout,
        UPDATE_KVCACHE=(k_cache is not None and v_cache is not None),
        SELECT_KVCACHE=(k_cache is not None and v_cache is not None),
        num_warps=num_warps,
        num_stages=1,
    )
    return q
