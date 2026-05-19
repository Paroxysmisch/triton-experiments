, :] + start_n) < N_CTX
        k = tl.load(K_ptrs + HEAD_DIM * offs_n[None, :] + start_n, mask=k_mask)
        qk = tl.dot(q, k, allow_tf32=False)
        if STAGE == 2:
            qk += tl.load(K_scale_ptr) * offs_m[:, None]
            tl.store(K_scale_ptr, tl.load(K_scale_ptr) * 2)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk * q_scale
        qk = tl.math.exp2(qk - m_ij[:, None])
        p = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 0)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        p = p.to(tl.float16)
        v = tl.load(V_ptrs + HEAD_DIM * offs_n[None, :] + start_n, mask=k_mask)
        acc += tl.dot(p, v, allow_tf32=False)
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, Out, Q_scale, K_scale, 
             Z, H, N_CTX, 
             HEAD_DIM: tl.constexpr, 
             BLOCK_M: tl.constexpr, 
             BLOCK_N: tl.constexpr, 
             STAGE: tl.constexpr, 
             offs_m: tl.constexpr, offs_n: tl.constexpr):
    # Compute block pointers
    q_ptr = Q + BLOCK_M * offs_m[:, None] * HEAD_DIM
    k_ptr = K + BLOCK_N * offs_n[None, :] * HEAD_DIM
    v_ptr = V + BLOCK_N * offs_n[None, :] * HEAD_DIM
    out_ptr = Out + BLOCK_M * offs_m[:, None] * HEAD_DIM
    c1 = tl.math.log2(tl.math.log2(N_CTX)) + 1.0
    q_scale = Q_scale * c1
    K_scale = tl.math.exp2(K_scale * c1)
    # initialize memory pointers to first block
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # loop over blocks
    for start_m in range(0, (N_CTX + BLOCK_M - 1) // BLOCK_M, STAGE):
        q = tl.load(q_ptr, mask=(offs_m[:, None] < N_CTX))
        qk_scale = q_scale
        q_scale *= 2
        if STAGE == 2:
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, qk_scale, k_ptr, K_scale, v_ptr,  
                                            start_m, BLOCK_M, HEAD_DIM, BLOCK_N, 
                                            STAGE, offs_m, offs_n, N_CTX)
        if STAGE == 1:
            tl.store(out_ptr, acc.to(tl.float16), mask=(offs_m[:, None] < N_CTX))
        q_ptr += BLOCK_M * HEAD_DIM
        k_ptr += BLOCK_N * HEAD_DIM
        v_ptr += BLOCK_N * HEAD_DIM
        out_ptr += BLOCK_M * HEAD_DIM
    # scale acc
    l_i += 1e-6
    alpha = tl.math.exp2(m_i) / l_i
    acc = acc * alpha[:, None]
    # write back last block
    if STAGE == 1:
        tl.store(out_ptr, acc.to(tl.float16), mask=(offs_m[:, None] < N_CTX))
    return

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q_scale, k_scale, is_causal, is_sliding_window):
        # run kernel
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        extra_kern_args = []
        if is_sliding_window:
            extra_kern_args = ['extra_arg']
        BLOCK_M = 128
        BLOCK_N = 64
        if HEAD_DIM_K > 64:
            BLOCK_N = 32
        if HEAD_DIM_K > 128:
            BLOCK_N = 16
            BLOCK_M = 64
        if HEAD_DIM_K > 256:
            BLOCK_N = 16
            BLOCK_M = 32
        full_block_m = (q.shape[2] + BLOCK_M - 1) // BLOCK_M
        STAGE_1_M = min(2, full_block_m)
        STAGE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048][q.shape[2] + BLOCK_M - 1]
        if q.shape[2] <= 48:
            BLOCK_M = 48
            BLOCK_N = 16
        if q.shape[2] <= 24:
            BLOCK_M = 24
            BLOCK_N = 16
        if q.shape[2] <= 12:
            BLOCK_M = 12
            BLOCK_N = 16
        if q.shape[2] <= 6:
            BLOCK_M = 6
            BLOCK_N = 16
        if q.shape[2] <= 3:
            BLOCK_M = 3
            BLOCK_N = 16
        if q.shape[2] <= 1:
            BLOCK_M = 1
            BLOCK_N = 16
        if q.shape[2] >= 1024:
            BLOCK_M = 1024
            BLOCK_N = 64
        if q.shape[2] >= 2048:
            BLOCK_M = 2048
            BLOCK_N = 64
        if q.shape[2] >= 4096:
            BLOCK_M = 4096
            BLOCK_N = 64
        if q.shape[2] >= 8192:
            BLOCK_M = 8192
            BLOCK_N = 64
        if q.shape[2] >= 16384:
            BLOCK_M = 16384
            BLOCK_N = 64
        if q.shape[2] >= 32768:
            BLOCK_M = 32768
            BLOCK_N = 64
        if q.shape[2] >= 65536:
            BLOCK_M = 65536
            BLOCK_N = 64
        def grid(META): return (triton.cdiv(q.shape[2], META['BLOCK_M']), q.shape[0] * q.shape[1], 1)
        _attn_fwd[(grid, extra_kern_args)](
            q, k, v, o, q_scale, k_scale, 
            q.shape[0], q.shape[1], q.shape[2], 
            HEAD_DIM_K, BLOCK_M, BLOCK_N, 
            STAGE, 
            q.trans(0, 2, 1).to(tl.pointer_type(tl.float16)).to(tl.float32),
            q.trans(0, 2, 1).to(tl.pointer_type(tl.float16)).to(tl.float32),
            N_CTX=q.shape[2],
            num_warps=4 if HEAD_DIM_K <= 64 else 8,
            num_stages=1,
        )
        return o
