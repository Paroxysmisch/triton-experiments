, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_i_new)
        beta = tl.math.exp2(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
            other=0.0,
        )

        p = p.to(V.dtype.element_ty)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


def context_attention_fwd(
    q, k, v, o, b_start_loc, b_seq_len, max_input_len, context_len, q_per_kv_head
):
    if isinstance(b_start_loc, torch.Tensor):
        b_start_loc = b_start_loc.contiguous()
    if isinstance(b_seq_len, torch.Tensor):
        b_seq_len = b_seq_len.contiguous()
    sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    kv_group_num = q.shape[1] // k.shape[1]

    if q.shape[2] > 1024:
        BLOCK = 512
    else:
        BLOCK = 128

    if q.dtype == torch.float16:
        num_warps = 8
    else:
        num_warps = 4

    batch, head = b_seq_len.shape[0], q.shape[1]
    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    # grid = (batch, head, triton.cdiv(context_len, BLOCK))

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=q.shape[2],
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
