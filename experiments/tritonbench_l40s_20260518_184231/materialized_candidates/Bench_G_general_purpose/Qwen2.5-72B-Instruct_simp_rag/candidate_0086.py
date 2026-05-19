import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q,
    K,
    V,
    B0,
    Out,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(Q.shape[2], BLOCK_M)
    num_pid_n = tl.cdiv(K.shape[2], BLOCK_N)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid = pid % num_pid_in_batch
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + (batch_id * stride_qb + tl.program_id(1) * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (batch_id * stride_kb + tl.program_id(1) * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None])
    v_ptrs = V + (batch_id * stride_vb + tl.program_id(1) * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :])
    b0_ptrs = B0 + (batch_id * stride_qb + tl.program_id(1) * stride_qh + offs_m[:, None] * stride_qm + offs_n[None, :])

    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)
    b0 = tl.load(b0_ptrs, mask=offs_m[:, None] < Q.shape[2], other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")

    for start_n in range(0, K.shape[2], BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.dot(q, k, allow_tf32=True)
        qk = qk * sm_scale + b0
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

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
        v = tl.load(v_ptrs + start_n * stride_vn)
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new

    out_ptrs = Out + (batch_id * stride_ob + tl.program_id(1) * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < Q.shape[2])

def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, out, sm_scale, BLOCK_M=128, BLOCK_N=128, BLOCK_DMODEL=64):
    assert q.shape[0] == k.shape[0] == v.shape[0] == b0.shape[0] == out.shape[0]
    assert q.shape[1] == k.shape[1] == v.shape[1] == b0.shape[1] == out.shape[1]
    assert q.shape[2] == k.shape[2] == v.shape[2] == b0.shape[2] == out.shape[2]
    assert q.shape[3] == k.shape[3] == v.shape[3] == out.shape[3]
    assert b0.shape[3] == q.shape[2]

    grid = (q.shape[2] // BLOCK_M, q.shape[0] * q.shape[1])
    num_warps = 4 if BLOCK_DMODEL <= 64 else 8

    _fwd_kernel_aligned[grid](
        q,
        k,
        v,
        b0,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=1,
    )
