import triton
import triton.language as tl

# Triton kernel for forward attention
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Q_scale,
    K_scale,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    # Initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch * N_CTX + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    off_k = offs_n[None, :] * stride_kbs + cur_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + cur_head * stride_vh + offs_d[None, :]

    q = tl.load(Q + off_q, mask=offs_m[:, None] < N_CTX, other=0.0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # Initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch * N_CTX + start_n) * stride_kbs,
            mask=(start_n + offs_n[None, :]) < N_CTX,
            other=0.0,
        )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= Q_scale * K_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
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
            v_ptrs + (cur_batch * N_CTX + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < N_CTX,
            other=0.0,
        )

        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # Initialize pointers to output
    off_o = (
        (cur_batch * N_CTX + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)

# Helper function for the inner loop
@triton.jit
def _attn_fwd_inner(
    q,
    k,
    v,
    q_scale,
    k_scale,
    m_i,
    l_i,
    acc,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # -- compute qk ----
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= q_scale * k_scale
    qk = tl.where(tl.arange(0, BLOCK_M)[:, None] >= tl.arange(0, BLOCK_N)[None, :], qk, float("-inf"))

    # -- compute m_ij, p, l_ij
    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    # -- update m_i and l_i
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    # -- update output accumulator --
    # scale p
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    # scale acc
    acc_scale = l_i / l_i_new * alpha
    acc = acc * acc_scale[:, None]
    # update acc
    p = p.to(v.dtype)
    acc += tl.dot(p, v)
    # update m_i and l_i
    l_i = l_i_new
    m_i = m_i_new
    return m_i, l_i, acc

# Wrapper function to configure and launch the kernel
def context_attention_fwd(q, k, v, o, n_ctx, block_m, block_dmodel, block_n):
    if CUDA_CAPABILITY[0] >= 8:
        BLOCK = 128
    else:
        BLOCK = 64

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}

    sm_scale = 1.0 / (Lq**0.5)
    batch, head = q.shape[0], q.shape[1]

    grid = (batch, head, triton.cdiv(n_ctx, BLOCK))
    num_warps = 4 if Lk <= 64 else 8

    _attn_fwd[grid](
        q,
        k,
        v,
        sm_scale,
        sm_scale,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        N_CTX=n_ctx,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )

# Example usage
# q, k, v, o = ... (initialize tensors)
# n_ctx = ... (context size)
# block_m, block_dmodel, block_n = 128, 64, 128
# context_attention_fwd(q, k, v, o, n_ctx, block_m, block_dmodel, block_n)
