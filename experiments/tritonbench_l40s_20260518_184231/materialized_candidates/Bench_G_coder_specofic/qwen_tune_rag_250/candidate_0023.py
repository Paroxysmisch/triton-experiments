).to(q_elem_type)
    sin_l = tl.load(SIN + cs_offset_l).to(q_elem_type)
    sin_h = tl.load(SIN + cs_offset_h).to(q_elem_type)

    if BLOCK_QH == 1:
        q_offset = (head_id * half_size + feat_offset_l)[None, :] \
            + tl.arange(0, BLOCK)[:, None] * stride_qd
        q_to_store = Q + q_offset
        e_offset = (head_id * half_size + feat_offset_l)[None, :] \
            + tl.arange(0, BLOCK)[:, None] * stride_qes
        e_to_store = Q_EMB + e_offset
        acc_q0 = tl.load(q_to_store).to(q_elem_type)
        acc_q1 = tl.load(q_to_store + half_size).to(q_elem_type)
        acc_q0 = acc_q0.to(tl.float32)
        acc_q1 = acc_q1.to(tl.float32)
        out_q0 = acc_q0 * cos_l - acc_q1 * sin_l
        out_q0 = out_q0.to(q_elem_type)
        tl.store(e_to_store, out_q0, mask=seq_mask)
        tl.store(e_to_store + half_size, out_q0, mask=seq_mask)

        out_q1 = acc_q0 * sin_l + acc_q1 * cos_h
        out_q1 = out_q1.to(q_elem_type)
        tl.store(e_to_store + half_size, out_q1, mask=seq_mask)

    else:
        k_offset = (head_id * half_size + feat_offset_l)[None, :] \
            + tl.arange(0, BLOCK)[:, None] * stride_kd
        k_to_store = K + k_offset
        e_offset = (head_id * half_size + feat_offset_l)[None, :] \
            + tl.arange(0, BLOCK)[:, None] * stride_kes
        e_to_store = K_EMB + e_offset
        acc_k0 = tl.load(k_to_store).to(q_elem_type)
        acc_k1 = tl.load(k_to_store + half_size).to(q_elem_type)
        acc_k0 = acc_k0.to(tl.float32)
        acc_k1 = acc_k1.to(tl.float32)
        out_k0 = acc_k0 * cos_l - acc_k1 * sin_l
        out_k0 = out_k0.to(q_elem_type)
        tl.store(e_to_store, out_k0, mask=seq_mask)
        tl.store(e_to_store + half_size, out_k0, mask=seq_mask)

        out_k1 = acc_k0 * sin_l + acc_k1 * cos_h
        out_k1 = out_k1.to(q_elem_type)
        tl.store(e_to_store + half_size, out_k1, mask=seq_mask)

def apply_rotary_pos_emb(q: Tensor,
                         k: Tensor,
                         cos: Tensor,
                         sin: Tensor,
                         q_embed: Tensor = None,
                         k_embed: Tensor = None):
    """Apply rotary position embedding on q and k.

    Args:
        q (Tensor): Query state.
        k (Tensor): Key state.
        cos (Tensor): cosine matrix (seq_len, dim).
        sin (Tensor): sine matrix (seq_len, dim).
        q_embed (Tensor, optional): output q. Defaults to None.
        k_embed (Tensor, optional): output k. Defaults to None.

    Returns:
        Tuple[Tensor, Tensor]: Embedded query and key.
    """
    if cos.device != q.device:
        cos = cos.to(device=q.device)
    if sin.device != q.device:
        sin = sin.to(device=q.device)

    seq_len = cos.numel() // cos.size(-1)
    BLOCK = 16
    BLOCK_N = 16
    half_size = q.size(-1) // 2
    BLOCK_QH = q.size(-2)

    if q_embed is None:
        q_embed = torch.empty_like(q)
    if k_embed is None:
        k_embed = torch.empty_like(k)

    grid = (triton.cdiv(seq_len, BLOCK), q.size(-2) * q.size(-3))
    apply_rotary_pos_emb_qk_kernel[grid](q,
                                         k,
                                         cos,
                                         sin,
                                         q_embed,
                                         k_embed,
                                         seq_len,
                                         stride_qs=q.stride(-3),
                                         stride_qh=q.stride(-2),
                                         stride_qd=q.stride(-1),
                                         stride_ks=k.stride(-3),
                                         stride_kh=k.stride(-2),
                                         stride_kd=k.stride(-1),
                                         stride_qes=q_embed.stride(-3),
                                         stride_qeh=q_embed.stride(-2),
                                         stride_qed=q_embed.stride(-1),
                                         stride_kes=k_embed.stride(-3),
                                         stride_keh=k_embed.stride(-2),
                                         stride_ked=k_embed.stride(-1),
                                         half_size=half_size,
                                         BLOCK=BLOCK,
                                         BLOCK_QH=BLOCK_QH,
                                         BLOCK_N=BLOCK_N,
                                         num_warps=4,
                                         num_stages=2)
    return q_embed, k_embed
