import triton
import triton.language as tl


@triton.jit
def decoding_fused_rotary_embedding_kernel(
    Q_PTR, K_PTR, V_PTR,
    K_CACHE_PTR, V_CACHE_PTR,
    SINE_PTR, COS_PTR,
    BLOCK_TABLES_PTR, KV_LENGTHS_PTR,
    q_stride_b, q_stride_h, q_stride_t, q_stride_d,
    k_stride_b, k_stride_h, k_stride_t, k_stride_d,
    v_stride_b, v_stride_h, v_stride_t, v_stride_d,
    kc_stride_g, kc_stride_b, kc_stride_h, kc_stride_t, kc_stride_d,
    vc_stride_g, vc_stride_b, vc_stride_h, vc_stride_t, vc_stride_d,
    batch_offset_q, head_offset_q, token_offset_q,
    batch_offset_k, head_offset_k, token_offset_k,
    batch_offset_v, head_offset_v, token_offset_v,
    block_tables_offset, kv_lengths_offset,
    current_batch, current_token,
    KV_GROUP_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    USE_NEW_KCACHE_LAYOUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)

    head_id = pid_h
    token_id = pid_t

    # Offsets for Q, K, V
    q_batch_idx = current_batch + batch_offset_q
    q_head_idx = head_id + head_offset_q
    q_token_idx = token_id + token_offset_q

    k_batch_idx = current_batch + batch_offset_k
    k_head_idx = head_id + head_offset_k
    k_token_idx = token_id + token_offset_k

    v_batch_idx = current_batch + batch_offset_v
    v_head_idx = head_id + head_offset_v
    v_token_idx = token_id + token_offset_v

    # Index range for this block in the head dimension
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < HEAD_DIM

    # Compute pointers
    q_offset = (q_batch_idx * q_stride_b
                + q_head_idx * q_stride_h
                + q_token_idx * q_stride_t)
    k_offset = (k_batch_idx * k_stride_b
                + k_head_idx * k_stride_h
                + k_token_idx * k_stride_t)
    v_offset = (v_batch_idx * v_stride_b
                + v_head_idx * v_stride_h
                + v_token_idx * v_stride_t)

    q_ptrs = Q_PTR + (q_offset + idx) * q_stride_d
    k_ptrs = K_PTR + (k_offset + idx) * k_stride_d
    v_ptrs = V_PTR + (v_offset + idx) * v_stride_d

    # Load Q, K, V
    q_data = tl.load(q_ptrs, mask=mask, other=0.0)
    k_data = tl.load(k_ptrs, mask=mask, other=0.0)
    v_data = tl.load(v_ptrs, mask=mask, other=0.0)

    # Sine/Cosine
    sine_ptrs = SINE_PTR + idx
    cos_ptrs = COS_PTR + idx
    sin_data = tl.load(sine_ptrs, mask=mask, other=0.0)
    cos_data = tl.load(cos_ptrs, mask=mask, other=0.0)

    # Split head dimension in two halves
    half_head = HEAD_DIM // 2
    half_mask = idx < half_head
    idx_half = idx + half_head

    # Rotary embedding on Q (two halves)
    q_data_first = tl.where(half_mask, q_data, 0.0)
    q_data_second = tl.where(half_mask, tl.load(q_ptrs + half_head * q_stride_d, mask=half_mask, other=0.0), 0.0)

    cos_first = tl.where(half_mask, cos_data, 0.0)
    sin_first = tl.where(half_mask, sin_data, 0.0)

    # x', y'
    x_rot = q_data_first * cos_first - q_data_second * sin_first
    y_rot = q_data_second * cos_first + q_data_first * sin_first

    # Write back Q
    tl.store(q_ptrs, x_rot, mask=half_mask)
    tl.store(q_ptrs + half_head * q_stride_d, y_rot, mask=half_mask)

    # Rotary embedding on K, V if needed and store in caches
    # We only do partial indexing if HEAD_DIM > 0, same approach
    k_data_first = tl.where(half_mask, k_data, 0.0)
    k_data_second = tl.where(half_mask, tl.load(k_ptrs + half_head * k_stride_d, mask=half_mask, other=0.0), 0.0)
    xk_rot = k_data_first * cos_first - k_data_second * sin_first
    yk_rot = k_data_second * cos_first + k_data_first * sin_first

    v_data_first = tl.where(half_mask, v_data, 0.0)
    v_data_second = tl.where(half_mask, tl.load(v_ptrs + half_head * v_stride_d, mask=half_mask, other=0.0), 0.0)
    xv_rot = v_data_first * cos_first - v_data_second * sin_first
    yv_rot = v_data_second * cos_first + v_data_first * sin_first

    # We can optionally store in K/V caches
    # kc_offset, vc_offset depend on layout
    # For demonstration, assume we partition by group -> (k_batch_idx // KV_GROUP_NUM)
    if USE_NEW_KCACHE_LAYOUT:
        cache_g_idx = head_id // KV_GROUP_NUM
        kc_offset = (cache_g_idx * kc_stride_g
                     + (k_batch_idx * kc_stride_b)
                     + ((head_id % KV_GROUP_NUM) * kc_stride_h)
                     + k_token_idx * kc_stride_t)
        vc_offset = (cache_g_idx * vc_stride_g
                     + (v_batch_idx * vc_stride_b)
                     + ((v_head_idx % KV_GROUP_NUM) * vc_stride_h)
                     + v_token_idx * vc_stride_t)
    else:
        kc_offset = (k_batch_idx * kc_stride_b
                     + head_id * kc_stride_h
                     + k_token_idx * kc_stride_t)
        vc_offset = (v_batch_idx * vc_stride_b
                     + v_head_idx * vc_stride_h
                     + v_token_idx * vc_stride_t)

    kc_ptrs = K_CACHE_PTR + (kc_offset + idx) * kc_stride_d
    vc_ptrs = V_CACHE_PTR + (vc_offset + idx) * vc_stride_d

    tl.store(kc_ptrs, xk_rot, mask=half_mask)
    tl.store(kc_ptrs + half_head * kc_stride_d, yk_rot, mask=half_mask)
    tl.store(vc_ptrs, xv_rot, mask=half_mask)
    tl.store(vc_ptrs + half_head * vc_stride_d, yv_rot, mask=half_mask)


def decoding_fused_rotary_embedding(
    q, k, v,
    k_cache, v_cache,
    sine, cos,
    block_tables, kv_lengths,
    current_batch, current_token,
    kv_group_num,
    use_new_kcache_layout=False
):
    head_dim = q.shape[-1]
    q_head_num = q.shape[1]
    q_total_tokens = q.shape[2]

    # Choose BLOCK_SIZE and warps
    BLOCK_SIZE = 128 if head_dim > 64 else 64
    num_warps = 4 if head_dim >= 64 else 1

    grid = (q_head_num, q_total_tokens)

    # Strides and offsets (example usage; actual values depend on data layout)
    q_strides = q.stride()
    k_strides = k.stride()
    v_strides = v.stride()
    kc_strides = k_cache.stride()
    vc_strides = v_cache.stride()

    batch_offset_q = 0
    head_offset_q = 0
    token_offset_q = 0
    batch_offset_k = 0
    head_offset_k = 0
    token_offset_k = 0
    batch_offset_v = 0
    head_offset_v = 0
    token_offset_v = 0

    block_tables_offset = 0
    kv_lengths_offset = 0

    decoding_fused_rotary_embedding_kernel[
        grid
    ](
        q, k, v,
        k_cache, v_cache,
        sine, cos,
        block_tables, kv_lengths,
        q_strides[0], q_strides[1], q_strides[2], q_strides[3],
        k_strides[0], k_strides[1], k_strides[2], k_strides[3],
        v_strides[0], v_strides[1], v_strides[2], v_strides[3],
        kc_strides[0], kc_strides[1], kc_strides[2], kc_strides[3], kc_strides[4] if len(kc_strides) > 4 else 1,
        vc_strides[0], vc_strides[1], vc_strides[2], vc_strides[3], vc_strides[4] if len(vc_strides) > 4 else 1,
        batch_offset_q, head_offset_q, token_offset_q,
        batch_offset_k, head_offset_k, token_offset_k,
        batch_offset_v, head_offset_v, token_offset_v,
        block_tables_offset, kv_lengths_offset,
        current_batch, current_token,
        KV_GROUP_NUM=kv_group_num,
        HEAD_DIM=head_dim,
        USE_NEW_KCACHE_LAYOUT=use_new_kcache_layout,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
