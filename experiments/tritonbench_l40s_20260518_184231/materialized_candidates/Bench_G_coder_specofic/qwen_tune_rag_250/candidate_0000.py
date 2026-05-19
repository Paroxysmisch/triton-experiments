h, s_yt, s_yn = Y.stride()
    _, s_zt, s_zn = Z.stride()
    _, s_hh, s_hm, s_hk = H.stride()
    q_ptrs = Q + off * s_qh + (s_qm * BLOCK_M + offs_m[:, None]) * s_qm + offs_d[None, :] * s_qk
    k_ptrs = K + off * s_kh + (offs_n[None, :] * s_kn + offs_d[:, None]) * s_kk
    v_ptrs = V + off * s_vh + (offs_n[:, None] * s_vk) * s_vk
    y_ptrs = Y + off * s_yh + (s_yt * BLOCK_M + offs_m[:, None]) * s_yt + offs_n[None, :] * s_yn
    z_ptrs = Z + (s_zt * BLOCK_M + offs_m[:, None]) * s_zt + offs_n[None, :] * s_zn
    h_ptrs = H + (off * s_hh + (s_hm * BLOCK_M + offs_m[:, None]) * s_hm + offs_d[None, :] * s_hk)

    lo = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    loop_k = tl.cdiv(N_CTX, BLOCK_K)
    for start_n in range(0, loop_k):
        start_n = tl.multiple_of(start_n, BLOCK_K // 16) * (BLOCK_K // 16)
        k = tl.load(
            k_ptrs + (start_n * s_kn),
            mask=(start_n + offs_n[None, :]) < N_CTX,
            other=0.0,
        )
        qk = tl.zeros([BLOCK_M, BLOCK_K // 16], dtype=tl.float32)
        q = tl.load(q_ptrs + (start_n * s_qn), mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where((start_n + offs_n[None, :]) < N_CTX, qk, float("-inf"))
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(
            v_ptrs + (start_n * s_vn),
            mask=(start_n + offs_n[:, None]) < N_CTX,
            other=0.0,
        )
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new

    lo = tl.where((offs_m[:, None] + off * BLOCK_M) < N_CTX, lo, 0)
    m = tl.where((offs_m + off * BLOCK_M) < N_CTX, m_i, float("-inf"))
    p = tl.exp(m_i - m)
    l = tl.sum(p, 0)
    p_scale = 1.0 / l
    p = p * p_scale
    lo = lo * p_scale[:, None]
    y = lo + acc
    z = p[:, None] * q

    m_ptrs = M + off * s_qh + (s_qm * BLOCK_M + offs_m[:, None]) * s_qm + offs_d[None, :] * s_qk
    l_ptrs = L + off * s_qh + (s_qm * BLOCK_M + offs_m[:, None]) * s_qm + offs_d[None, :] * s_qk
    tl.store(m_ptrs, m)
    tl.store(l_ptrs, l)
    tl.store(y_ptrs, y.to(Y.dtype.element_ty))
    tl.store(z_ptrs, z.to(Z.dtype.element_ty))
    tl.debug_barrier()
    h = tl.load(h_ptrs)
    u = tl.exp(m - m_i_new)
    v = tl.load(
        v_ptrs + ((start_n + 1) * s_vn),
        mask=((start_n + 1) + offs_n[:, None]) < N_CTX,
        other=0.0,
    )
    z = tl.load(z_ptrs + ((start_n + 1) * s_zn), mask=((start_n + 1) + offs_n[:, None]) < N_CTX)
    q = tl.load(
        q_ptrs + ((start_n + 1) * s_qn),
        mask=((start_n + 1) + offs_n[:, None]) < N_CTX,
        other=0.0,
    )
    k = tl.load(
        k_ptrs + ((start_n + 1) * s_kn),
        mask=((start_n + 1) + offs_n[None, :]) < N_CTX,
        other=0.0,
    )
    k = k.to(v.dtype)
    v = v.to(q.dtype)
    u = u.to(q.dtype)
    h = h.to(q.dtype)
    z_new = u * z
    acc = tl.dot(p, v)
    lo = h * z_new[:, None]
    qk = tl.dot(q, k)
    qk *= sm_scale
    qk = tl.where((start_n + offs_n[None, :]) < N_CTX, qk, float("-inf"))
    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    p_scale = tl.exp(m_i - m_i_new)
    l_i_new = p_scale * l_i + l_ij * z_new
    acc_scale = p_scale * l_i / l_i_new
    acc = acc * acc_scale[:, None]
    lo = lo + acc
    m = tl.maximum(m_i, m_i_new)
    p_scale = tl.exp(m_i - m)
    l = p_scale * l_i + l_ij * z_new
    p = p * p_scale
    m_ptrs = M + off * s_qh + (s_qm * BLOCK_M + offs_m[:, None]) * s_qm + offs_d[None, :] * s_qk
    l_ptrs = L + off * s_qh + (s_qm * BLOCK_M + offs_m[:, None]) * s_qm + offs_d[None, :] * s_qk
    tl.store(m_ptrs, m)
    tl.store(l_ptrs, l)
    y_ptrs = Y + off * s_yh + (s_yt * BLOCK_M + offs_m[:, None]) * s_yt + offs_n[None, :] * s_yn
    tl.store(y_ptrs, lo.to(Y.dtype.element_ty))


@triton.jit
def _bwd_prep(
    Y,
    DY,
    L,
    NewDY,
    Delta,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off = tl.program_id(0)
    start = tl.program_id(1)
    offs_m = start * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, D_HEAD)
    y_ptrs = Y + off * BLOCK_M + offs_m
    dy_ptrs = DY + off * BLOCK_M + offs_m
    newdy_ptrs = NewDY + off * BLOCK_M + offs_m
    l_ptrs = L + off * BLOCK_M + offs_m
    delta_ptrs = Delta + offs_m

    m = tl.load(l_ptrs)
    dy = tl.load(dy_ptrs)
    y = tl.load(y_ptrs)
    denom = tl.load(delta_ptrs)
    newdy = dy * m / denom
    tl.store(newdy_ptrs, newdy)
    tl.debug_barrier()
    # Update Q grad
    qk = tl.load(y_ptrs, mask=offs_m[:, None] < offs_n[None, :], other=0.0)
    qk = tl.sum(qk * newdy, axis=1)
    tl.store(delta_ptrs, qk)


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Y,
    DY,
    DQ,
    DK,
    DV,
    L,
    M,
    D,
    Z,
    H,
    N_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    off = tl.program_id(0)
    start = tl.program_id(1)
    s_qk = Q.stride(2)
    s_qn = Q.stride(1)
    s_vn = V.stride(1)
    s_qh = Q.stride(0)
    s_vh = V.stride(0)
    s
