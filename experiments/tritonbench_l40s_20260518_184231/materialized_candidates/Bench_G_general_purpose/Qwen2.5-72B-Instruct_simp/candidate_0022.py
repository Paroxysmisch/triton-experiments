import triton
import triton.language as tl

# Constants
BLOCK = 128

@triton.jit
def _fwd_kernel(Q, K, V, Out, L, M, sm_scale, stride_qm, stride_kn, stride_vm, stride_vn, stride_om, stride_on, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_heads = Q.shape[0]
    head_id = pid % num_heads
    block_id = pid // num_heads

    # Pointers to Q, K, V, and Out
    Q += head_id * stride_qm + block_id * BLOCK_M
    K += head_id * stride_kn
    V += head_id * stride_vm
    Out += head_id * stride_om + block_id * BLOCK_M

    # Pointers to L and M
    L += head_id * (Out.shape[1] // BLOCK_M) + block_id
    M += head_id * (Out.shape[1] // BLOCK_M) + block_id

    # Offsets for blocks
    offsets_m = tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_DMODEL)

    # Load Q and V
    q = tl.load(Q + offsets_m[:, None] * stride_qm + offsets_k[None, :] * BLOCK_DMODEL)
    v = tl.load(V + offsets_n[None, :] * stride_vn + offsets_k[:, None] * BLOCK_DMODEL)

    # Initialize L and M
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')

    for start_n in range(0, K.shape[1], BLOCK_N):
        # Load K
        k = tl.load(K + start_n * stride_kn + offsets_k[:, None] * BLOCK_DMODEL + offsets_n[None, :] * BLOCK_N)

        # Compute QK^T
        qk = tl.dot(q, k, allow_tf32=True)
        qk = qk * sm_scale

        # Compute attention scores
        qk = qk + tl.where(m_i[:, None] >= qk, 0, m_i[:, None] - qk)
        m_i = tl.max(m_i, tl.max(qk, 1))

        # Compute L
        l_i_new = tl.logsumexp(qk, 1)
        l_i = l_i + tl.exp(l_i_new - l_i)

        # Compute softmax
        p = tl.exp(qk - m_i[:, None])

        # Compute Out
        out = tl.dot(p, v, allow_tf32=True)
        tl.store(Out + start_n * stride_on + offsets_m[:, None] * stride_om + offsets_n[None, :], out)

    # Store L and M
    tl.store(L, l_i)
    tl.store(M, m_i)

@triton.jit
def _bwd_kernel(Q, K, V, Out, GradOut, GradQ, GradK, GradV, L, M, sm_scale, stride_qm, stride_kn, stride_vm, stride_vn, stride_om, stride_on, stride_gqm, stride_gkn, stride_gvm, stride_gvn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_heads = Q.shape[0]
    head_id = pid % num_heads
    block_id = pid // num_heads

    # Pointers to Q, K, V, Out, GradOut, GradQ, GradK, GradV
    Q += head_id * stride_qm + block_id * BLOCK_M
    K += head_id * stride_kn
    V += head_id * stride_vm
    Out += head_id * stride_om + block_id * BLOCK_M
    GradOut += head_id * stride_on + block_id * BLOCK_M
    GradQ += head_id * stride_gqm + block_id * BLOCK_M
    GradK += head_id * stride_gkn
    GradV += head_id * stride_gvm

    # Pointers to L and M
    L += head_id * (Out.shape[1] // BLOCK_M) + block_id
    M += head_id * (Out.shape[1] // BLOCK_M) + block_id

    # Offsets for blocks
    offsets_m = tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_DMODEL)

    # Load Q and V
    q = tl.load(Q + offsets_m[:, None] * stride_qm + offsets_k[None, :] * BLOCK_DMODEL)
    v = tl.load(V + offsets_n[None, :] * stride_vn + offsets_k[:, None] * BLOCK_DMODEL)

    # Load L and M
    l_i = tl.load(L)
    m_i = tl.load(M)

    for start_n in range(0, K.shape[1], BLOCK_N):
        # Load K
        k = tl.load(K + start_n * stride_kn + offsets_k[:, None] * BLOCK_DMODEL + offsets_n[None, :] * BLOCK_N)

        # Compute QK^T
        qk = tl.dot(q, k, allow_tf32=True)
        qk = qk * sm_scale

        # Compute attention scores
        qk = qk + tl.where(m_i[:, None] >= qk, 0, m_i[:, None] - qk)
        p = tl.exp(qk - m_i[:, None])

        # Load GradOut
        grad_out = tl.load(GradOut + start_n * stride_on + offsets_m[:, None] * stride_om + offsets_n[None, :])

        # Compute GradV
        grad_v = tl.dot(p, grad_out, allow_tf32=True)
        tl.atomic_add(GradV + start_n * stride_gvn + offsets_n[None, :] * stride_gvm + offsets_k[:, None] * BLOCK_DMODEL, grad_v)

        # Compute GradQ
        grad_q = tl.dot(grad_out, v, allow_tf32=True)
        tl.atomic_add(GradQ + offsets_m[:, None] * stride_gqm + offsets_k[None, :] * BLOCK_DMODEL, grad_q)

        # Compute GradK
        grad_k = tl.dot(q, grad_out, allow_tf32=True)
        tl.atomic_add(GradK + start_n * stride_gkn + offsets_k[:, None] * BLOCK_DMODEL + offsets_n[None, :] * BLOCK_N, grad_k)

    # Compute GradL
    grad_l = tl.load(GradOut + offsets_m[:, None] * stride_om + offsets_n[None, :])
    grad_l = tl.sum(grad_l, 1)
    tl.atomic_add(L, grad_l)
