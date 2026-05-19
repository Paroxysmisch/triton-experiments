import triton
import triton.language as tl

# Constants
BLOCK = 128
BLOCK_M = BLOCK
BLOCK_N = BLOCK
BLOCK_DMODEL = BLOCK

# Forward Kernel
@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    L, m,
    stride_qm, stride_kn, stride_vm, stride_om,
    stride_qk, stride_kk, stride_vk, stride_ok,
    nheads, N_CTX,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    head_id = pid // (N_CTX // BLOCK_M)
    block_id = pid % (N_CTX // BLOCK_M)
    offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_offsets = (bid * stride_qm + head_id * stride_qk + offs_m[:, None] * stride_qm + offs_d[None, :])
    k_offsets = (bid * stride_kn + head_id * stride_kk + offs_n[None, :] * stride_kn + offs_d[:, None])
    v_offsets = (bid * stride_vm + head_id * stride_vk + offs_n[:, None] * stride_vm + offs_d[None, :])
    out_offsets = (bid * stride_om + head_id * stride_ok + offs_m[:, None] * stride_om + offs_d[None, :])

    q = tl.load(Q + q_offsets)
    k = tl.load(K + k_offsets)
    v = tl.load(V + v_offsets)

    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float('inf')

    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K + k_offsets + start_n * BLOCK_N * stride_kn)
        qk = tl.dot(q, k, allow_tf32=True)
        qk = qk * sm_scale
        qk = tl.where(offs_m[:, None] >= (offs_n[None, :] + start_n), qk, float('-inf'))
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_i_new[:, None])
        l_i_new = l_i + tl.sum(p, 1)
        p = p / l_i_new[:, None]
        acc += tl.dot(p, v, allow_tf32=True)
        l_i = l_i_new
        m_i = m_i_new

    acc = acc / l_i[:, None]
    tl.store(Out + out_offsets, acc)
    tl.store(L + bid * N_CTX + block_id * BLOCK_M + offs_m, l_i)
    tl.store(m + bid * N_CTX + block_id * BLOCK_M + offs_m, m_i)

# Backward Preprocess
@triton.jit
def _bwd_preprocess(
    DO, L, Delta,
    stride_dom, stride_dl,
    N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    block_id = pid % (N_CTX // BLOCK_M)
    offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_DMODEL)
    do_offsets = (bid * stride_dom + offs_m[:, None] * stride_dom + offs_n[None, :])
    l_offsets = (bid * stride_dl + offs_m)
    do = tl.load(DO + do_offsets)
    l = tl.load(L + l_offsets)
    delta = tl.sum(do * do, axis=1)
    delta = delta / l
    tl.store(Delta + bid * N_CTX + block_id * BLOCK_M + offs_m, delta)

# Backward Kernel
@triton.jit
def _bwd_kernel(
    Q, K, V, DO, DQ, DK, DV,
    L, m, Delta,
    stride_qm, stride_kn, stride_vm, stride_dom,
    stride_qk, stride_kk, stride_vk, stride_dqm, stride_dkn, stride_dvk,
    nheads, N_CTX,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    head_id = pid // (N_CTX // BLOCK_M)
    block_id = pid % (N_CTX // BLOCK_M)
    offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_offsets = (bid * stride_qm + head_id * stride_qk + offs_m[:, None] * stride_qm + offs_d[None, :])
    k_offsets = (bid * stride_kn + head_id * stride_kk + offs_n[None, :] * stride_kn + offs_d[:, None])
    v_offsets = (bid * stride_vm + head_id * stride_vk + offs_n[:, None] * stride_vm + offs_d[None, :])
    do_offsets = (bid * stride_dom + head_id * stride_dqk + offs_m[:, None] * stride_dom + offs_d[None, :])
    dq_offsets = (bid * stride_dqm + head_id * stride_dqk + offs_m[:, None] * stride_dqm + offs_d[None, :])
    dk_offsets = (bid * stride_dkn + head_id * stride_dqk + offs_n[None, :] * stride_dkn + offs_d[:, None])
    dv_offsets = (bid * stride_dvk + head_id * stride_dqk + offs_n[:, None] * stride_dvk + offs_d[None, :])

    q = tl.load(Q + q_offsets)
    k = tl.load(K + k_offsets)
    v = tl.load(V + v_offsets)
    do = tl.load(DO + do_offsets)
    delta = tl.load(Delta + bid * N_CTX + block_id * BLOCK_M + offs_m)
    l = tl.load(L + bid * N_CTX + block_id * BLOCK_M + offs_m)
    m = tl.load(m + bid * N_CTX + block_id * BLOCK_M + offs_m)

    dq = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    dk = tl.zeros((BLOCK_DMODEL, BLOCK_N), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_DMODEL), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K + k_offsets + start_n * BLOCK_N * stride_kn)
        v = tl.load(V + v_offsets + start_n * BLOCK_N * stride_vm)
        qk = tl.dot(q, k, allow_tf32=True)
        qk = qk * sm_scale
        qk = tl.where(offs_m[:, None] >= (offs_n[None, :] + start_n), qk, float('-inf'))
        p = tl.exp(qk - m[:, None])
        p = p / l[:, None]
        do = tl.load(DO + do_offsets + start_n * BLOCK_N * stride_dom)
        dv += tl.dot(p, do, allow_tf32=True)
        dp = tl.dot(do, v, allow_tf32=True)
        dp = dp * p
        dp = dp - tl.sum(dp, 1)[:, None] * p
        dq += tl.dot(dp, k, allow_tf32=True)
        dk += tl.dot(q, dp, allow_tf32=True)

    dq = dq * delta[:, None]
    tl.store(DQ + dq_offsets, dq)
    tl.store(DK + dk_offsets, dk)
    tl.store(DV + dv_offsets, dv)
