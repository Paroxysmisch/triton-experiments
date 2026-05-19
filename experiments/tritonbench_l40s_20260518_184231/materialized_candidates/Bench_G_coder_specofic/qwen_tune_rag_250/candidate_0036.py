+ rk[None, :] * stride_ctx_k)

    for d_max_offset in range(d_model, 0, -BLOCK_D):
        q_tile = tl.load(q_ptr_tile, mask=rd[None, :] < d_max_offset, other=0.0)
        k_tile = tl.load(k_ptr_tile, mask=rd[:, None] < d_max_offset, other=0.0)
        acc_tile += tl.dot(q_tile, k_tile)
        q_ptr_tile += BLOCK_D * stride_d
        k_ptr_tile += BLOCK_D * stride_d

    acc_tile = acc_tile.to(scores_ptr.dtype.element_ty)

    rq = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    scores_offset_tile = rq[:, None] * stride_out_q + rk[None, :] * stride_out_k
    scores_ptr_tile = scores_ptr + scores_offset_tile

    mask = (rq < n_ctx_q)[:, None] & (rk < n_ctx_k)[None, :]
    tl.store(scores_ptr_tile, acc_tile, mask=mask)


def qk_dotprod(q, k):
    # q: [batch, seq, n_heads, head_size]
    # k: [batch, seq, n_heads, head_size]
    assert q.shape[-1] == k.shape[-1]
    assert q.shape[-2] > 0 and q.shape[-2] <= 16384
    assert q.is_contiguous()

    batch, seq_q, n_heads, head_size = q.shape
    _, seq_k, _, _ = k.shape

    scores = torch.empty(
        (batch, n_heads, seq_q, seq_k), dtype=torch.float16, device=q.device
    )

    assert (
        q.device == scores.device
    ), "Input and scores must be on the same device before calling qk_dotprod"

    # grid = lambda META: (
    #     triton.cdiv(seq_q, META["BLOCK_Q"]) * triton.cdiv(seq_k, META["BLOCK_K"]),
    # )
    stride_d = head_size - 1
    stride_ctx_q = q.stride(-2)
    stride_ctx_k = k.stride(-2)
    stride_out_q = scores.stride(-2)
    stride_out_k = scores.stride(-1)

    _kernel[(seq_q, seq_k)](
        q,
        k,
        scores,
        seq_q,
        seq_k,
        head_size,
        stride_ctx_q,
        stride_ctx_k,
        stride_d,
        stride_out_q,
        stride_out_k,
    )
    return scores
