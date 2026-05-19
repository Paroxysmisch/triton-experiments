[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        p = p.to(tl.float16)
        v = tl.load(V_ptrs, mask=k_mask)
        acc += tl.dot(p, v)
        K_ptrs += BLOCK_N
        V_ptrs += BLOCK_N
        K_scale_ptr += 1

@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out, 
              stride_qz, stride_qh, stride_qm, stride_qk, 
              stride_kz, stride_kh, stride_kn, stride_kk, 
              stride_vz, stride_vh, stride_vk, stride_vn, 
              stride_oz, stride_oh, stride_om, stride_on, 
              Z, H, N_CTX, HEAD_DIM, 
              BLOCK_M, BLOCK_N, STAGE: tl.constexpr):
    # shape constraints
    stride_qm = stride_qm
    stride_qk = stride_qk
    stride_kn = stride_kn
    stride_kk = stride_kk
    stride_vk = stride_vk
    stride_vn = stride_vn
    stride_qh = stride_qh
    stride_kh = stride_kh
    stride_vh = stride_vh
    stride_oh = stride_oh
    stride_qz = stride_qz
    stride_kz = stride_kz
    stride_vz = stride_vz
    pidhm = tl.program_id(0)
    pidz = tl.program_id(1)
    pidh = tl.program_id(2)
    # create index ranges
    offs_m = (pidhm * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    # initialize offsets
    offs_q = pidz * stride_qz + pidh * stride_qh + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk
    offs_k = offs_n[None, :] * stride_kn + offs_n[:, None] * stride_kk
    offs_v = offs_n[None, :] * stride_vk + offs_n[:, None] * stride_vn
    # initialize pointer to m and l -> load from provided pointer
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q_scale = tl.load(Q_scale + pidz * stride_qz + pidh * stride_qh)
    q = tl.load(Q + offs_q, mask=(offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX), other=0.0)
    # causal mask
    causal_mask = (offs_m[:, None] >= (offs_n[None, :]))
    if STAGE & 1:
        _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K + offs_k, K_scale + pidz * stride_kz + pidh * stride_kh, V + offs_v, 0, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX)
    if STAGE & 2:
        _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K + offs_k, K_scale + pidz * stride_kz + pidh * stride_kh, V + offs_v, N_CTX - BLOCK_N, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX)
    # scale acc
    l_i = 1.0 / l_i
    acc = acc * l_i[:, None]
    # write back l and m
    m_ptrs = None
    l_ptrs = None
    # write back acc
    offs_o = pidz * stride_qz + pidh * stride_qh + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk
    tl.store(Out + offs_o, acc, mask=(offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX))
    return

class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q_scale, k_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        # broadcast shapes
        q_shape = q.shape
        k_shape = k.shape
        v_shape = v.shape
        q = q.reshape(-1, q.shape[-2], q.shape[-1])
        k = k.reshape(-1, k.shape[-2], k.shape[-1])
        v = v.reshape(-1, v.shape[-2], v.shape[-1])
        o = o.reshape(-1, o.shape[-2], o.shape[-1])
        batch, head, seq_len, head_dim = q.shape
        # work out stage
        stage = 0
        if seq_len % 2 == 0:
            stage |= 1
        if seq_len > BLOCK_M:
            stage |= 2
        # enqueue kernel
        grid = (triton.cdiv(seq_len, BLOCK_M), batch, head)
        ctx.size_guards = (seq_len <= 4096, head_dim <= 128)
        _attn_fwd[grid](q, k, v, q_scale, k_scale, o, 
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3), 
                        k.stride(0), k.stride(1), k.stride(2), k.stride(3), 
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3), 
                        o.stride(0), o.stride(1), o.stride(2), o.stride(3), 
                        batch, head, seq_len, head_dim, 
                        BLOCK_M, BLOCK_N, stage)
        ctx.save_for_backward(q, k, v, o, q_scale, k_scale)
        o = o.reshape(q_shape)
        q = q.reshape(q_shape)
        k = k.reshape(k_shape)
        v = v.reshape(v_shape)
        return o, q, k, v

attention = Attention.apply
