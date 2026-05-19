+ batch_id)
    k_seqlen = tl.load(KVSeqLens + batch_id)
    block_offset = tl.load(BlockOffsets + block_id)

    # for each token, copy one block
    for token_id in range(q_startloc, q_startloc + q_seqlen, BLOCK):
        copy_num = tl.minimum(q_startloc + q_seqlen - token_id, BLOCK)
        d_off = tl.arange(0, copy_num)

        # token offset
        token_off = token_id + d_off
        # copy k cache
        k_val = tl.load(
            KStates + token_off[:, None] * stride_kss + stride_ksh * h_off[:, None],
            mask=token_off[:, None] < k_seqlen,
        )
        tl.store(
            KCaches + block_offset * stride_kcn + token_off[:, None] * stride_kcb,
            k_val,
            mask=token_off[:, None] < k_seqlen,
        )
        # copy v cache
        v_val = tl.load(
            VStates + token_off[:, None] * stride_vss + stride_vsh * h_off[:, None],
            mask=token_off[:, None] < k_seqlen,
        )
        tl.store(
            VCaches + block_offset * stride_vcn + token_off[:, None] * stride_vcb,
            v_val,
            mask=token_off[:, None] < k_seqlen,
        )

def fill_kv_cache(
    k_states: Tensor,
    v_states: Tensor,
    k_caches: Tensor,
    v_caches: Tensor,
    q_start_loc: Tensor,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
    block_offsets: Tensor,
    k_scale_zeros: Tensor = None,
    v_scale_zeros: Tensor = None,
    quant_policy: int = 0,
):
    """fill key, value cache."""
    if quant_policy == 0:
        kv_dim = v_states.shape[-1]
        major, minor = v_states.shape[:-2], v_states.shape[-2:]
        kv_states = v_states.reshape((np.prod(major), *minor))
        major, minor = k_states.shape[:-2], k_states.shape[-2:]
        kv_states = torch.cat([kv_states, k_states.reshape((np.prod(major), *minor))], dim=-1)
        major, minor = k_caches.shape[:-3], k_caches.shape[-3:]
        kv_caches = k_caches.reshape((np.prod(major), *minor))
        major, minor = v_caches.shape[:-3], v_caches.shape[-3:]
        kv_caches = torch.cat([kv_caches, v_caches.reshape((np.prod(major), *minor))], dim=-4)

        kv_states = kv_states.contiguous()
        kv_caches = kv_caches.contiguous()
        q_seq_len = q_seq_len.contiguous()
        b, h, n, d = kv_states.shape
        BLOCK = 32
        BLOCK_D = _div_up(d, BLOCK)
        BLOCK_H = min(65536 // (b * h * 128), 64)
        grid = (b, BLOCK_D)
        meta = get_kernel_meta(kv_states)
        _fill_kv_cache_kernel[grid](
            kv_states,
            kv_states,
            kv_caches,
            kv_caches,
            q_start_loc,
            q_seq_len,
            kv_seq_len,
            block_offsets,
            num_heads=h,
            head_dim=d,
            head_dim_v=kv_dim,
            stride_kss=kv_states.stride(-3),
            stride_ksh=kv_states.stride(-2),
            stride_ksd=kv_states.stride(-1),
            stride_vss=kv_states.stride(-3),
            stride_vsh=kv_states.stride(-2),
            stride_vsd=kv_states.stride(-1),
            stride_kcn=kv_caches.stride(-4),
            stride_kcb=kv_caches.stride(-3),
            stride_kch=kv_caches.stride(-2),
            stride_kcd=kv_caches.stride(-1),
            stride_vcn=kv_caches.stride(-3),
            stride_vcb=kv_caches.stride(-2),
            stride_vch=kv_caches.stride(-1),
            stride_vcd=kv_caches.stride(0),
            stride_boff=block_offsets.stride(0),
            BLOCK=BLOCK,
            BLOCK_D=BLOCK_D,
            BLOCK_DV=_div_up(kv_dim, BLOCK),
            BLOCK_H=BLOCK_H,
            **meta,
        )
        kv_caches = kv_caches.reshape(major + (-3, BLOCK_H, BLOCK_D, BLOCK))
        k_caches, v_caches = torch.tensor_split(kv_caches, 2, dim=-4)
        return k_caches, v_caches
    else:
        major, minor = v_states.shape[:-2], v_states.shape[-2:]
        kv_states = v_states.reshape((np.prod(major), *minor))
        major, minor = k_states.shape[:-2], k_states.shape[-2:]
        kv_states = torch.cat([kv_states, k_states.reshape((np.prod(major), *minor))], dim=-1)
        q_seq_len = q_seq_len.contiguous()
        b, h, n, d = kv_states.shape
        BLOCK = 32
        BLOCK_D = _div_up(d, BLOCK)
        grid = (b,)
        meta = get_kernel_meta(kv_states)
        _fill_kv_cache_quant_kernel[grid](
            kv_states,
            kv_states,
            kv_states,
            kv_states,
            q_start_loc,
            q_seq_len,
            kv_seq_len,
            block_offsets,
            k_scale_zeros,
            v_scale_zeros,
            num_heads=h,
            head_dim=d,
            head_dim_v=d,
            stride_kss=kv_states.stride(-3),
            stride_ksh=kv_states.stride(-2),
            stride_ksd=kv_states.stride(-1),
            stride_vss=kv_states.stride(-3),
            stride_vsh=kv_states.stride(-2),
            stride_vsd=kv_states.stride(-1),
            stride_kcn=kv_states.stride(-3),
            stride_kcb=kv_states.stride(-2),
            stride_kch=kv_states.stride(-1),
            stride_kcd=kv_states.stride(0),
            stride_vcn=kv_states.stride(-3),
            stride_vcb=kv_states.stride(-2),
            stride_vch=kv_states.stride(-1),
            stride_vcd=kv_states.stride(1),
            stride_boff=block_offsets.stride(0),
            BLOCK=BLOCK,
            BLOCK_D=BLOCK_D,
            BLOCK_DV=_div_up(d, BLOCK),
            **meta,
        )
        kv_states = kv_states.reshape(major + (-2, BLOCK_D))
        k_states, v_states = torch.tensor_split(kv_states, 2, dim=-3)
        return k_states, v_states

@triton.jit
def _fill_kv_cache_quant_kernel(
    KStates,
    VStates,
    KScalesZeros,
    KVStates,
    QStartLoc,
    QSeqLens,
    KVSeqLens,
    BlockOffsets,
    KScaleZeros,
    VScaleZeros,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    head_dim_v: tl.constexpr,
    stride_kss,
    stride_ksh,
    stride_ksd,
    stride_vss,
    stride_vsh,
    stride_vsd,
    stride_kcn: tl.constexpr,
    stride_kcb: tl.constexpr,
    stride_kch: tl.constexpr,
    stride_kcd: tl.constexpr,
    stride_vcn: tl.constexpr,
    stride_vcb: tl.constexpr,
    stride_vch: tl.constexpr,
    stride_vcd: tl.constexpr,
    stride_boff,
    stride_ksc,
    stride_kzc,
    stride_vsc,
    stride_vzc,
    BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """fill kv cache kernel."""
    batch_id = tl.program_id(0)

    # initialize
    h_off = tl.arange(0, BLOCK_H)
    d_off = tl.arange(0, BLOCK_D)

    q_startloc = tl.load(QStartLoc + batch_id)
    q_seqlen = tl.load(QSeqLens + batch_id)
    k_seqlen = tl.load(KVSeqLens + batch_id)
    block_offset = tl.load(BlockOffsets + batch_id)

    # for each token, copy one block
    for token_id in range(q_startloc, q_startloc + q_seqlen, BLOCK):
        copy_num = tl.minimum(q_startloc + q_seqlen - token_id, BLOCK)
        d_off = tl.arange(0, copy_num)

        # token offset
        token_off = token_id + d_off
        # copy k cache
        k_val = tl.load(
            KStates + token_off[:, None] * stride_kss + stride_ksh * h_off[:, None],
            mask=token_off[:, None] < k_seqlen,
        )
        k_scales = tl.load(
            KScalesZeros + token_off[:, None] * stride_ksc + stride_kzc * h_off[:, None],
            mask=token_off[:, None] < k_seqlen,
        )
        k_scale, k_zero = _quant_int4(k_val, k_scales)
        tl.store(
            KVStates + block_offset * stride_kcn + stride_kcd * d_off[:, None] + token_off[:, None] * stride_kcb,
            k
