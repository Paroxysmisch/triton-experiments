= tl.arange(0, KCACHE_X)
    offsets_dmodel_k = cur_kv_head_idx * stride_kh + range_x * stride_kd
    offsets_dmodel_v = cur_kv_head_idx * stride_vh + range_x * stride_vd

    # Triton 2.1.0
    # Only support the case when the head dim of KV is the same
    # The kernel will load keys or values from different sequences into a different block
    # It will not load data from different sequences into the same block
    offsets_k = offsets_in_last_block * stride_kt + offsets_dmodel_k
    k = tl.load(K + offsets_k)
    offsets_kcache = (
        block_id * stride_kcb
        + offsets_dmodel_k
        + offsets_in_last_block * stride_kcs
        + range_x * stride_kcd
    )
    tl.store(KCache + offsets_kcache, k)

    offsets_v = offsets_in_last_block * stride_vt + offsets_dmodel_v
    v = tl.load(V + offsets_v)
    offsets_vcache = block_id * stride_vcb + offsets_dmodel_v + offsets_in_last_block * stride_vcs
    tl.store(VCache + offsets_vcache, v)
    return


def copy_k_to_blocked_cache(K, KCache, BLOCK_TABLES, seq_len, max_seq_len, stride_kt, stride_kh, stride_kd, head_dim,
                            head_dim_kv, block_size, x):
    # assert K and KCache have the same shape
    assert K.shape[1] == KCache.shape[1] and K.shape[2] == KCache.shape[-1]
    n_kv_heads = K.shape[1]
    grid = (max_seq_len, n_kv_heads, 1)
    if KCache.shape[-1] == head_dim_kv:
        # [num_blocks, num_kv_heads, head_dim // x, block_size, x]
        grid = (max_seq_len, n_kv_heads, head_dim // x)
    _copy_to_kcache_seqlen_n_kernel[grid](
        K,
        KCache,
        BLOCK_TABLES,
        seq_len,
        stride_kt,
        stride_kh,
        stride_kd,
        KCache.stride(0),
        KCache.stride(1),
        KCache.stride(2) if KCache.shape[-1] == head_dim_kv else KCache.stride(3),
        KCache.stride(3) if KCache.shape[-1] == head_dim_kv else KCache.stride(4),
        KCache.stride(1) if KCache.shape[-1] == head_dim_kv else KCache.stride(2),
        KCache.stride(0),
        BLOCK_TABLES.stride(0),
        BLOCK_TABLES.stride(1),
        block_size,
        max_seq_len,
        HEAD_DIM=head_dim,
        KCACHE_X=x,
    )
    return


def copy_kv_to_blocked_cache(K, V, KCache, VCache, BLOCK_TABLES, context_length, stride_kt, stride_kh, stride_kd,
                             stride_vt, stride_vh, stride_vd, head_dim, head_dim_kv, block_size, x):
    # assert K, V, KCache, VCache have the same shape
    assert K.shape[1] == V.shape[1] == KCache.shape[1] and K.shape[2] == V.shape[2] and K.shape[2] == KCache.shape[-1]
    n_kv_heads = K.shape[1]
    grid = (1, n_kv_heads, 1)
    if KCache.shape[-1] == head_dim_kv:
        # [num_blocks, num_kv_heads, head_dim // x, block_size, x]
        grid = (1, n_kv_heads, head_dim // x)
    _copy_to_kvcache_seqlen1_kernel[grid](
        K,
        V,
        KCache,
        VCache,
        BLOCK_TABLES,
        context_length,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vt,
        stride_vh,
        stride_vd,
        KCache.stride(0),
        KCache.stride(1),
        KCache.stride(2) if KCache.shape[-1] == head_dim_kv else KCache.stride(3),
        KCache.stride(3) if KCache.shape[-1] == head_dim_kv else KCache.stride(4),
        KCache.stride(1) if KCache.shape[-1] == head_dim_kv else KCache.stride(2),
        KCache.stride(0),
        VCache.stride(0),
        VCache.stride(1),
        VCache.stride(2),
        VCache.stride(1),
        VCache.stride(0),
        BLOCK_TABLES.stride(0),
        BLOCK_TABLES.stride(1),
        block_size,
        HEAD_DIM=head_dim,
        KCACHE_X=x,
    )
    return
