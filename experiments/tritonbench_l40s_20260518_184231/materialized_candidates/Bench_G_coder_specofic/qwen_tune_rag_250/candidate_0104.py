_kn, stride_kk,  
              stride_vz, stride_vh, stride_vk, stride_vn,  
              stride_oz, stride_oh, stride_om, stride_on,  
              Z, H, N_CTX, HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  
              STAGE: tl.constexpr):
    tl.static_assert(BLOCK_N <= HEAD_DIM, 'BLOCK_N must be less than or equal to HEAD_DIM')
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    Q_ptrs = Q + qvk_offset
    V_ptrs = V + qvk_offset
    K_ptrs = K + qvk_offset
    Out_ptrs = Out + qvk_offset

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q_scale = Q_scale[off_hz]
    q = tl.load(Q_ptrs + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk, 
                mask=(offs_m[:, None] < N_CTX) & (offs_n[None, :] < HEAD_DIM),
                other=0.0)

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale, V_ptrs, 
                                        start_m, BLOCK_M, HEAD_DIM, BLOCK_N, offs_m, offs_n, N_CTX,  
                                        1, N_CTX, HEAD_DIM, BLOCK_N, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale, V_ptrs, 
                                        start_m, BLOCK_M, HEAD_DIM, BLOCK_N, offs_m, offs_n, N_CTX,  
                                        2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    acc = acc / l_i[:, None]
    m_i = tl.maximum(m_i, tl.max(tl.math.log2(l_i), 0))
    acc = acc * tl.math.exp2(m_i)

    o = tl.dot(tl.math.exp2(m_i.to(tl.float32)), acc.to(tl.float16), out_dtype=tl.float16)

    o_ptrs = Out_ptrs + stride_om * offs_m[:, None] + stride_on * offs_n[None, :]
    o_mask = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < HEAD_DIM)
    tl.store(o_ptrs, o, mask=o_mask)

def attn_fwd(q, k, v, q_scale, k_scale):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 3
    if k_scale is None:
        k_scale = tl.math.rsqrt(tl.math.abs(tl.sum(tl.math.square(k), axis=2)).to(tl.float32) + 1e-6)
    if q_scale is None:
        q_scale = 1.0 / tl.math.sqrt(tl.math.abs(tl.sum(tl.math.square(q), axis=2)).to(tl.float32) + 1e-6)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    o = torch.empty_like(q)
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    ctx = q.shape[2]
    def grid(meta): return (triton.cdiv(meta['N_CTX'], meta['BLOCK_M']), meta['Z'] * meta['H'], 1)
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], ctx,
        HEAD_DIM=q.shape[-1], 
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        STAGE=stage, num_warps=1, num_stages=1)
    return o, k_scale
