_scale_ptr += 1
        V_ptrs += BLOCK_N * HEAD_DIM

@triton.jit
def _attn_fwd(Q, K, V, K_scale, V_scale, Out, 
            stride_qz, stride_qh, stride_qm, stride_qk, 
            stride_kz, stride_kh, stride_kn, stride_kk, 
            stride_vz, stride_vh, stride_vk, stride_vn,
            stride_oz, stride_oh, stride_om, stride_on,
            Z, H, N_CTX, HEAD_DIM: tl.constexpr, 
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
            STAGE: tl.constexpr, 
            M_CTX: tl.constexpr, 
            N_CTX: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    Q_ptrs = Q + qvk_offset + \
        (BLOCK_M * start_m + tl.arange(0, BLOCK_M))[:, None] * stride_qm + \
        tl.arange(0, BLOCK_N)[None, :] * stride_qk

    V_ptrs = V + qvk_offset + \
        tl.arange(0, BLOCK_N)[:, None] * stride_vk + \
        tl.arange(0, BLOCK_M)[None, :] * stride_vm

    K_ptrs = K + qvk_offset + \
        tl.arange(0, BLOCK_N)[None, :] * stride_kk + \
        (BLOCK_M * start_m + tl.arange(0, BLOCK_M)[:, None]) * stride_kn

    K_scale_ptrs = K_scale + \
        (BLOCK_M * start_m + tl.arange(0, BLOCK_M)[:, None]) + \
        tl.arange(0, BLOCK_N)[None, :] * stride_kk

    O_ptrs = Out + qvk_offset + \
        (BLOCK_M * start_m + tl.arange(0, BLOCK_M)[:, None]) * stride_om + \
        tl.arange(0, BLOCK_N)[None, :] * stride_on

    offs_m = BLOCK_M * start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < N_CTX,
                other=0.0).to(tl.float16)
    q = (q / tl.math.sqrt(HEAD_DIM)).to(tl.float32)
    q_scale = tl.math.exp2(q * 0).to(tl.float32)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptrs, V_ptrs,
                    start_m, BLOCK_M, HEAD_DIM, BLOCK_N, offs_m, offs_n, N_CTX,
                    STAGE=STAGE, M_CTX=M_CTX, N_CTX=N_CTX)

    m_i_last = tl.maximum(m_i, tl.max(acc, 1))
    p = tl.math.exp2(acc - m_i_last[:, None])
    l_i = l_i * tl.math.exp2(m_i - m_i_last) + tl.sum(p, 1)
    out = tl.dot(p.to(tl.float16), V, out_dtype=tl.float16)
    out = out / l_i[:, None]

    m_i = m_i_last
    V_ptrs += BLOCK_N * HEAD_DIM
    K_scale_ptrs += BLOCK_N
    start_m += 1

    for start_m in range(start_m, tl.cdiv(N_CTX, BLOCK_M)):
        _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptrs, V_ptrs,
                        start_m, BLOCK_M, HEAD_DIM, BLOCK_N, offs_m, offs_n, N_CTX,
                        STAGE=STAGE, M_CTX=M_CTX, N_CTX=N_CTX)

        m_i_last = tl.maximum(m_i, tl.max(acc, 1))
        p = tl.math.exp2(acc - m_i_last[:, None])
        l_i = l_i * tl.math.exp2(m_i - m_i_last) + tl.sum(p, 1)
        update = tl.dot(p.to(tl.float16), V, out_dtype=tl.float16)
        out += update
        out_scale = tl.math.exp2((m_i - m_i_last) * 0)
        out = out / out_scale[:, None]
        V_ptrs += BLOCK_N * HEAD_DIM
        K_scale_ptrs += BLOCK_N
        m_i = m_i_last
        start_m += 1

    m_i_last = tl.maximum(m_i, tl.max(acc, 1))
    out = out * tl.math.exp2(m_i - m_i_last)[:, None]
    out_scale = tl.math.exp2(m_i_last * 0)
    tl.store(O_ptrs, out / out_scale[:, None],
             mask=offs_m[:, None] < N_CTX)
