q_ptr = q + (offs_m[:, None] * stride_q + offs_n[None, :] * HEAD_DIM)
        q = tl.load(q_ptr)
        q = (q * q_scale).to(tl.float16)
        q = q.to(tl.float32)
    elif STAGE == 2:
        k_ptr = K_ptrs + (offs_m[:, None] * stride_k + offs_n[None, :] * HEAD_DIM)
        k = tl.load(k_ptr)
        k = (k * tl.load(K_scale_ptr)).to(tl.float32)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        l_ij = tl.sum(tl.exp(qk), 1)
    elif STAGE == 3:
        m_ij = tl.maximum(m_i, tl.load(m_ptrs + offs_m))
        q = tl.load(q + offs_m * stride_q)
        q = (q * q_scale).to(tl.float16)
        q = q.to(tl.float32)
    l_ij = tl.load(l_ptrs + offs_m) * tl.exp(m_i - m_ij) + tl.exp(l_ij) * tl.exp(m_ij - m_i)
    p = tl.exp(qk - m_ij[:, None])
    v = tl.load(V_ptrs + (offs_m[:, None] * stride_v + offs_n[None, :] * HEAD_DIM))
    acc = acc * tl.exp(m_i - m_ij)[:, None] + tl.dot(p.to(tl.float16), v)
    m_i = m_ij
    l_i = l_ij
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out,
            stride_qz, stride_qh, stride_qm, stride_qk, 
            stride_kz, stride_kh, stride_kn, stride_kk,
            stride_vz, stride_vh, stride_vk, stride_vn,
            stride_oz, stride_oh, stride_om, stride_on,
            Z, H, N_CTX, HEAD_DIM, BLOCK_M, BLOCK_N, STAGE, 
            num_stages, num_warps, 
            ):
    # 1. pre-compute offsets for matrix accesses
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # 2. initialize pointers to value vectors
    V_ptrs = V + (offs_m[:, None] * stride_vk + offs_n[None, :] * stride_vn)
    # 3. initialize pointer to first scaled key and set up cumulative sum
    K_ptrs = K
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    # 4. load q in a tile
    q = tl.load(Q + offs_m * stride_qm)
    q_scale = tl.load(Q_scale + offs_m)[..., None]
    # 5. pre-compute scaling factor used by all stages
    q_scale = q_scale * 1.44269504
    # 6. execute the different stages in the computation pipeline
    m_ptrs = tl.make_block_ptr(None, (N_CTX,), (1,), (0,), (1,), (0,))
    l_ptrs = tl.make_block_ptr(None, (N_CTX,), (1,), (0,), (1,), (0,))
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # stage 1: update keys
        _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale, V_ptrs, start_n, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX)
        # stage 2: update cumulative max and linear layer
        m_i, l_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale, V_ptrs, start_n, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE+1, offs_m, offs_n, N_CTX)
        # stage 3: read values
        m_i = tl.maximum(m_i, tl.load(m_ptrs + start_n)[offs_m])
        p = tl.exp((acc * l_i) * 1.44269504 - m_i[:, None])
        acc = acc * tl.exp((l_i - tl.load(l_ptrs + start_n)) * 1.44269504)[..., None]
        tl.store(Out + offs_m * stride_om + (start_n + offs_n) * stride_on, p.to(tl.float16))
        # update pointers
        K_ptrs = K_ptrs + BLOCK_N * stride_kn
        V_ptrs = V_ptrs + BLOCK_N * stride_vk
        m_ptrs = m_ptrs + BLOCK_N
        l_ptrs = l_ptrs + BLOCK_N
    return

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q_scale, k_scale):
        BLOCK = 128
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        # reshape input data into 2D tensors
        q2 = q.view(-1, q.shape[-1])
        k2 = k.view(-1, k.shape[-1])
        v2 = v.view(-1, v.shape[-1])
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // q.shape[-1]
        BLOCK = min(MAX_FUSED_SIZE, BLOCK)
        # heuristics for number of warps
        num_warps = min(max(BLOCK // 256, 1), 8)
        grid = (triton.cdiv(q2.shape[0], BLOCK), q2.shape[1], 1)
        # enqueue kernel
        _attn_fwd[grid](
            q2, k2, v2, q_scale, k_scale, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q2.shape[0], q2.shape[1], k2.shape[0], Lk,
            BLOCK, BLOCK, 3, num_warps=num_warps, num_stages=1)
        ctx.save_for_backward(q, k, v, o, q_scale, k_scale)
        return o

def attention(q, k, v, q_scale, k_scale):
    return _attention.apply(q, k, v, q_scale, k_scale)
