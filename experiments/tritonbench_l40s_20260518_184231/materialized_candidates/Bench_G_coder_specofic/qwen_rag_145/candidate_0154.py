@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q,
    k,
    v,
    cos,
    sin,
    k_cache,
    v_cache,
    BLOCK_TABLES,
    context_lengths,
    x,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_stride,
    kcb_stride,
    kch_stride,
    kcsplit_x_stride,
    kcs_stride,
    kcd_stride,
    vcb_stride,
    vch_stride,
    vcs_stride,
    vcd_stride,
    bts_stride,
    btb_stride,
    block_size,
    KV_GROUP_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_head = tl.program_id(1)
    ptx = pid_token * block_size + tl.program_id(2)

    if ptx < context_lengths[pid_head]:
        head_dim_offset = pid_head * head_dim_stride
        head_dim_offset += ptx
        cos_offset = pid_token * cos_token_stride
        cos_offset += head_dim_offset * cos_stride
        sin_offset = pid_token * cos_token_stride
        sin_offset += head_dim_offset * cos_stride
        q_offset = pid_head * q_head_stride
        q_offset += ptx
        k_offset = pid_head * k_head_stride
        k_offset += ptx
        v_offset = pid_head * k_head_stride
        v_offset += ptx
        k_cache_offset = pid_head * kcb_stride
        k_cache_offset += ptx * kch_stride
        v_cache_offset = pid_head * vcb_stride
        v_cache_offset += ptx * vch_stride

        cos_value = tl.load(cos + cos_offset)
        sin_value = tl.load(sin + sin_offset)

        for hid in range(HEAD_DIM // 2):
            q_idx = head_dim_offset + hid
            k_idx = head_dim_offset + hid
            v_idx = head_dim_offset + hid

            q_rotated = q[q_offset + q_idx * q_token_stride] * cos_value[hid] + q[q_offset + q_idx * q_token_stride + 1] * sin_value[hid]
            k_rotated = k[k_offset + k_idx * k_token_stride] * cos_value[hid] + k[k_offset + k_idx * k_token_stride + 1] * sin_value[hid]
            v_rotated = v[v_offset + v_idx * k_token_stride] * cos_value[hid] + v[v_offset + v_idx * k_token_stride + 1] * sin_value[hid]

            tl.store(q + q_offset + q_idx * q_token_stride, q_rotated)
            tl.store(k + k_offset + k_idx * k_token_stride, k_rotated)
            tl.store(v + v_offset + v_idx * k_token_stride, v_rotated)

            if k_cache is not None:
                tl.store(k_cache + k_cache_offset + hid * kcs_stride + ptx * kcd_stride, k_rotated)
            if v_cache is not None:
                tl.store(v_cache + v_cache_offset + hid * vcs_stride + ptx * vcd_stride, v_rotated)


def decoding_fused_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    kv_lengths: Optional[torch.Tensor] = None,
    use_new_kcache_layout: bool = False,
):
    # setup
    q_total_tokens, q_head_num, head_dim = q.shape

    # parameters for kernel
    head_dim_stride = q.stride(2)
    q_token_stride = q.stride(0)
    q_head_stride = q.stride(1)
    k_token_stride = k.stride(0)
    k_head_stride = k.stride(1)

    cos_token_stride = cos.stride(0)
    cos_stride = cos.stride(1)

    kcb_stride, kch_stride, kcs_stride, kcd_stride = 0, 0, 0, 0
    vcb_stride, vch_stride, vcs_stride, vcd_stride = 0, 0, 0, 0

    block_size = q_total_tokens // q_head_num

    if k_cache is not None:
        kcb_stride = k_cache.stride(0)
        kch_stride = k_cache.stride(1)
        kcs_stride = k_cache.stride(2)
        kcd_stride = k_cache.stride(3)
    if v_cache is not None:
        vcb_stride = v_cache.stride(0)
        vch_stride = v_cache.stride(1)
        vcs_stride = v_cache.stride(2)
        vcd_stride = v_cache.stride(3)

    # Kernel execution
    grid = (q_head_num, q_total_tokens // block_size)
    decoding_fused_rotary_embedding_kernel[grid](
        q,
        k,
        v,
        cos,
        sin,
        k_cache,
        v_cache,
        block_tables,
        kv_lengths,
        head_dim,
        q_token_stride,
        q_head_stride,
        k_token_stride,
        k_head_stride,
        head_dim_stride,
        cos_token_stride,
        cos_stride,
        kcb_stride,
        kch_stride,
        kcsplit_x_stride,
        kcs_stride,
        kcd_stride,
        vcb_stride,
        vch_stride,
        vcs_stride,
        vcd_stride,
        block_tables.stride(0),
        block_tables.stride(1),
        block_size,
        KV_GROUP_NUM=q_head_num // k.size(1),
        HEAD_DIM=head_dim,
        num_warps=16 if head_dim >= 512 else 8 if head_dim >= 256 else 4,
    )
