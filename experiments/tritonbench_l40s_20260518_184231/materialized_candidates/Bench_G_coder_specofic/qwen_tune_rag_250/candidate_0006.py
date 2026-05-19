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

        p = p.to(v.dtype)
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
    q,
    k,
    v,
    o,
    b_start_loc,
    b_seq_len,
    max_input_len,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    qkv_group_num: int = 1,
    block_size: int = 64,
):
    if triton.__version__ >= "2.1.0":
        kv_group_num = 1
    else:
        assert qkv_group_num == 1

    if triton.__version__ >= "2.1.0":
        BLOCK = block_size
    else:
        BLOCK = 128

    sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}
    batch, head = b_seq_len.shape[0], q.shape[1]
    # number of warps
    num_warps = max(BLOCK // q.shape[-1], 1)

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        stride_qbs,
        stride_qh,
        stride_kbs,
        stride_kh,
        stride_vbs,
        stride_vh,
        stride_obs,
        stride_oh,
        kv_group_num=1,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
