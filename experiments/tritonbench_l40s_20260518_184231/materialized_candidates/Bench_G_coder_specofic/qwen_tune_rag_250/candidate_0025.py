[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen_k,
                            other=0.0)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn,
                            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                            other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != 'none':
            if BIAS_TYPE == 'vector':
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == 'matrix':
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs + start_n,
                                   mask=(offs_m[:, None] < seqlen_q)
                                        & ((start_n + offs_n)[None, :] < seqlen_k),
                                   other=0.0).to(tl.float32)
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n)[:, None] < seqlen_k,
                            other=0.0)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn,
                            mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                            other=0.0)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    o_scale = tl.exp(m_i - lse_i)
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(out_ptrs, acc_o,
                     mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))


def _flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None):
    b, h, q_len, d = q.shape
    _, _, k_len, _ = k.shape
    if softmax_scale is None:
        softmax_scale = d ** -0.5
    assert q.shape == (b, h, q_len, d)
    assert k.shape == (b, h, k_len, d)
    assert v.shape == (b, h, k_len, d)
    if bias is not None:
        assert bias.shape == (b, h, q_len, k_len)
    assert (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and
            (bias is None or bias.is_contiguous()))
    BLOCK_HEADDIM = max(16, triton.next_power_of_2(d))
    num_warps = 4 if d < 64 else 8
    num_warps = 2
    if d >= 128:
        num_warps = 4
    if d >= 256:
        num_warps = 8
    if d >= 512:
        num_warps = 16
    if d >= 1024:
        num_warps = 32
    if d >= 2048:
        num_warps = 64
    BLOCK_M = 128
    while (BLOCK_M * BLOCK_M * 2 * d > 256 * 1024 * 1024):
        BLOCK_M //= 2
    seqlen_q_rounded = triton.cdiv(q_len, BLOCK_M) * BLOCK_M
    lse = torch.empty((b, h, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)
    grid = lambda META: (triton.cdiv(q_len, META["BLOCK_M"]), b * h, 1)
    _fwd_kernel[grid](
        q, k, v, bias, o,
        lse, torch.empty_like(lse),
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        bias.stride(0), bias.stride(1), bias.stride(2) if bias is not None else 0,
        o.stride(0), o.stride(1), o.stride(2),
        q.shape[1], q.shape[2], k.shape[2], seqlen_q_rounded, q.shape[-1],
        q_len, k_len,
        b, h,
        "vector" if bias is not None and bias.stride(3) == 1 else
        ("matrix" if bias is not None else "none"),
        causal, BLOCK_HEADDIM, BLOCK_M, BLOCK_N=num_warps * BLOCK_M,
        num_warps=num_warps,
        num_stages=3 if d < 256 else 2, )
    if q_len % BLOCK_M > 0:
        offs_q = q_len - q_len % BLOCK_M
        offs_m = offs_q + tl.arange(0, BLOCK_M)
        lse_ptrs = lse[:, :, offs_q:].reshape(-1, BLOCK_M)
        lse_i = tl.max(lse_ptrs, 1)
        m_ptrs = q + b * h * q_len * d + offs_q * d + tl.arange(0, d)
        o_ptrs = o + b * h * q_len * d + offs_q * d + tl.arange(0, d)
        if d < 256:
            num_warps = 4
        else:
            num_warps = 8
        if d >= 512:
            num_warps = 16
        if d >= 1024:
            num_warps = 32
        if d >= 2048:
            num_warps = 64
        _fwd_kernel[grid](
            q, k, v, bias, o,
            lse, torch.empty_like(lse),
            softmax_scale,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            bias.stride(0), bias.stride(1), bias.stride(2) if bias is not None else 0,
            o.stride(0
