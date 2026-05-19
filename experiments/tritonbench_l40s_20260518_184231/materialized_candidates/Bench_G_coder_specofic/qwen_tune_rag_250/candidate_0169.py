_X):
        offsets_k = (
            cur_kv_head_idx * stride_kh
            + split_x * stride_kcsplit_x
            + offsets_dmodel_x_partition * stride_kd
        )
        offsets_v = (
            cur_kv_head_idx * stride_vh
            + split_x * stride_kcsplit_x
            + offsets_dmodel_x_partition * stride_vd
        )
        offsets_kcache = (
            block_id * stride_kcb
            + offsets_k
            + offsets_in_last_block * stride_kcs
            + range_x * stride_kcd
        )
        offsets_vcache = (
            block_id * stride_vcb
            + offsets_v
            + offsets_in_last_block * stride_vcs
            + range_x * stride_vcd
        )
        k = tl.load(K + offsets_k)
        v = tl.load(V + offsets_v)
        tl.store(KCache + offsets_kcache, k)
        tl.store(VCache + offsets_vcache, v)
    return


def copy_k_to_blocked_cache(k, block_tables, k_cache, seq_len_cache, max_seq_len, use_new_cache_layout: bool):
    """
    Copy key tensor to the blocked cache.
    """
    if k.dim() == 4:
        # [num_tokens, num_kv_heads, head_dim // x, x]
        _, num_kv_heads, head_dim_over_x, x = k.shape
        num_tokens, _, _, _ = k_cache.shape
        assert x == 64 and "Triton only supports 64-dimensional partitions of key/value vectors"
        k = k.contiguous()
        k_cache = k_cache.contiguous()
        seq_len_cache = seq_len_cache.contiguous()
        assert (
            k_cache.size(1) == num_kv_heads
            and k_cache.size(2) == head_dim_over_x
            and k_cache.size(3) == x
        ), "The shape of key cache must be [num_blocks, num_kv_heads, head_dim // 64, 64]"
        assert (
            k.size(1) == num_kv_heads
            and k.size(2) == head_dim_over_x
            and k.size(3) == x
        ), "The shape of key must be [num_tokens, num_kv_heads, head_dim // 64, 64]"
        grid = (triton.cdiv(max_seq_len, 1), num_kv_heads, 64)
        kwargs = [
            k,
            k_cache,
            block_tables,
            seq_len_cache,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
        ]
        if use_new_cache_layout:
            kwargs += [k_cache.stride(4)]
        kwargs += [
            k_cache.stride(1),
            block_tables.stride(1),
            block_tables.stride(0),
            k_cache.size(3),
            min(triton.next_power_of_2(max_seq_len), 1024),
            HEAD_DIM=head_dim_over_x * x,
        ]
        if not use_new_cache_layout:
            kwargs += [k_cache.stride(2)]
        _copy_to_kcache_seqlen_n_kernel[grid](*kwargs)
    else:
        # [num_tokens, num_kv_heads, head_dim]
        num_tokens, num_kv_heads, head_dim = k.shape
        num_blocks, num_kv_heads, head_dim_over_x, block_size, x = k_cache.shape
        assert head_dim == head_dim_over_x * x
        assert (
            k_cache.size(1) == num_kv_heads
            and k_cache.size(2) == head_dim_over_x
            and k_cache.size(3) == block_size
            and k_cache.size(4) == x
        ), "The shape of key cache must be [num_blocks, num_kv_heads, head_dim // 64, block_size, 64]"
        k = k.contiguous().view(num_tokens, num_kv_heads, head_dim_over_x, x)
        k_cache = k_cache.contiguous()
        seq_len_cache = seq_len_cache.contiguous()
        assert (
            k.size(1) == num_kv_heads
            and k.size(2) == head_dim_over_x
            and k.size(3) == x
        ), "The shape of key must be [num_tokens, num_kv_heads, head_dim // 64, 64]"
        grid = (num_tokens, num_kv_heads, 64)
        kwargs = [
            k,
            k_cache,
            block_tables,
            seq_len_cache,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            k_cache.stride(4),
        ]
        if not use_new_cache_layout:
            kwargs += [k_cache.stride(2)]
        kwargs += [
            block_tables.stride(1),
            block_tables.stride(0),
            block_size,
            min(num_tokens, max_seq_len),
            HEAD_DIM=head_dim_over_x * x,
            KCACHE_X=x,
        ]
        _copy_to_kcache_seqlen_n_kernel[grid](*kwargs)
    return


def copy_kv_to_blocked_cache(
    k,
    v,
    block_tables,
    k_cache,
    v_cache,
    context_len_cache,
    max_context_len,
    use_new_cache_layout: bool,
):
    """
    Copy key and value tensors into the blocked cache.
    """
    if k.dim() == 4:
        # [num_tokens, num_kv_heads, head_dim // x, x]
        _, num_kv_heads, head_dim_over_x, x = k.shape
        num_tokens, _, _, _ = k_cache.shape
        assert x == 64 and "Triton only supports 64-dimensional partitions of key/value vectors"
        k = k.contiguous()
        v = v.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        context_len_cache = context_len_cache.contiguous()
        assert (
            k_cache.size(1) == num_kv_heads
            and k_cache.size(2) == head_dim_over_x
            and k_cache.size(3) == x
        ), "The shape of key cache must be [num_blocks, num_kv_heads, head_dim // 64, 64]"
        assert (
            v_cache.size(1) == num_kv_heads
            and v_cache.size(2) == head_dim_over_x
            and v_cache.size(3) == x
        ), "The shape of value cache must be [num_blocks, num_kv_heads, head_dim // 64, 64]"
        assert (
            k.size(1) == num_kv_heads
            and k.size(2) == head_dim_over_x
            and k.size(3) == x
        ), "The shape of key must be [num_tokens, num_kv_heads, head_dim // 64, 64]"
        assert (
            v.size(1) == num_kv_heads
            and v.size(2) == head_dim_over_x
            and v.size(3) == x
        ), "The shape of value must be [num_tokens, num_kv_heads, head_dim // 64, 64]"
        grid = (triton.cdiv(max_context_len, 1), num_kv_heads, 64)
        _copy_to_kvcache_seqlen1_kernel[grid](
            k,
            v,
            k_cache,
            v_cache,
            block_tables,
            context_len_cache,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            k_cache.stride(4),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            block_tables.stride(1),
            block_tables.stride(0),
            block_size=block_tables.shape[-1],
            HEAD_DIM=head_dim_over_x * x,
        )
    else:
        # [num_tokens, num_kv_heads, head_dim]
        num_tokens, num_kv_heads, head_dim = k.shape
        num_blocks, num_kv_heads, head_dim_over_x, block_size, x = k_cache.shape
        assert head_dim == head_dim_over_x * x
        assert (
            k_cache.size(1) == num_kv_heads
            and k_cache.size(2) == head_dim_over_x
            and k_cache.size(3) == block_size
            and k_cache.size(4) == x
        ), "The shape of key cache must be [num_blocks, num_kv_heads, head_dim // 64, block_size, 64]"
        assert (
            v_cache.size(1) == num_kv_heads
            and v_cache.size(2) == head_dim_over_x
            and v_cache.size(3) == block_size
            and v_cache.size(4) == x
        ), "The shape of value cache must be [num_blocks, num_kv_heads, head_dim // 64, block_size, 64]"
        k = k.contiguous().view(num_tokens, num_kv_heads, head_dim_over_x, x)
        v = v.contiguous().view(num_tokens, num_kv_heads, head_dim_over_x,
