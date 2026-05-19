import triton
import triton.language as tl

# Constants
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32

# Triton kernel for the forward pass of an attention operation
@triton.jit
def _attn_fwd(
    Q, K, V, o,
    q_scale, k_scale,
    stride_qm, stride_qk, stride_qv,
    stride_km, stride_kk, stride_kv,
    stride_om, stride_ok, stride_ov,
    M, N, K,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    m = pid // grid_n
    n = pid % grid_n
    q_off = m * stride_qm + n * stride_qn
    k_off = m * stride_km + n * stride_kn
    v_off = m * stride_vm + n * stride_vn
    o_off = m * stride_om + n * stride_on

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_off_i = k * stride_kk
        qk_off = q_off + k_off_i
        qk_ptr = Q + qk_off

        qk = tl.load(qk_ptr, mask=tl.arange(K) < K, other=0.0)
        qk = qk * q_scale * k_scale

        for i in range(0, BLOCK_M, 32):
            q_off_i = i * stride_qk
            q_ptr = Q + q_off + q_off_i
            q = tl.load(q_ptr, mask=tl.arange(BLOCK_M) < BLOCK_M, other=0.0)

            for j in range(0, BLOCK_N, 32):
                k_off_j = j * stride_kk
                k_ptr = K + k_off + k_off_j
                k = tl.load(k_ptr, mask=tl.arange(BLOCK_N) < BLOCK_N, other=0.0)

                qk = qk + q * k
                m_i = tl.maximum(m_i, qk)
                qk = qk - m_i
                exp_qk = tl.exp(qk)
                acc = acc + exp_qk * V + l_i
                l_i = l_i + exp_qk

        acc = acc / l_i
        o_off_i = i * stride_ok
        o_ptr = o + o_off + o_off_i
        tl.store(o_ptr, acc, mask=tl.arange(BLOCK_M) < BLOCK_M)

# Triton kernel for the inner loop of the attention operation
@triton.jit
def _attn_fwd_inner(
    Q, K, V, o,
    q_scale, k_scale,
    stride_qm, stride_qk, stride_qv,
    stride_km, stride_kk, stride_kv,
    stride_om, stride_ok, stride_ov,
    M, N, K,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    m = pid // grid_n
    n = pid % grid_n
    q_off = m * stride_qm + n * stride_qn
    k_off = m * stride_km + n * stride_kn
    v_off = m * stride_vm + n * stride_vn
    o_off = m * stride_om + n * stride_on

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_off_i = k * stride_kk
        qk_off = q_off + k_off_i
        qk_ptr = Q + qk_off

        qk = tl.load(qk_ptr, mask=tl.arange(K) < K, other=0.0)
        qk = qk * q_scale * k_scale

        for i in range(0, BLOCK_M, 32):
            q_off_i = i * stride_qk
            q_ptr = Q + q_off + q_off_i
            q = tl.load(q_ptr, mask=tl.arange(BLOCK_M) < BLOCK_M, other=0.0)

            for j in range(0, BLOCK_N, 32):
                k_off_j = j * stride_kk
                k_ptr = K + k_off + k_off_j
                k = tl.load(k_ptr, mask=tl.arange(BLOCK_N) < BLOCK_N, other=0.0)

                qk = qk + q * k
                m_i = tl.maximum(m_i, qk)
                qk = qk - m_i
                exp_qk = tl.exp(qk)
                acc = acc + exp_qk * V + l_i
                l_i = l_i + exp_qk

        acc = acc / l_i
        o_off_i = i * stride_ok
        o_ptr = o + o_off + o_off_i
        tl.store(o_ptr, acc, mask=tl.arange(BLOCK_M) < BLOCK_M)

# Triton kernel for the forward pass of an attention operation
@triton.jit
def attn_fwd(
    Q, K, V, o,
    q_scale, k_scale,
    stride_qm, stride_qk, stride_qv,
    stride_km, stride_kk, stride_kv,
    stride_om, stride_ok, stride_ov,
    M, N, K,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    m = pid // grid_n
    n = pid % grid_n
    q_off = m * stride_qm + n * stride_qn
    k_off = m * stride_km + n * stride_kn
    v_off = m * stride_vm + n * stride_vn
    o_off = m * stride_om + n * stride_on

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_off_i = k * stride_kk
        qk_off = q_off + k_off_i
        qk_ptr = Q + qk_off

        qk = tl.load(qk_ptr, mask=tl.arange(K) < K, other=0.0)
        qk = qk * q_scale * k_scale

        for i in range(0, BLOCK_M, 32):
            q_off_i = i * stride_qk
            q_ptr = Q + q_off + q_off_i
            q = tl.load(q_ptr, mask=tl.arange(BLOCK_M) < BLOCK_M, other=0.0)

            for j in range(0, BLOCK_N, 32):
                k_off_j = j * stride_kk
                k_ptr = K + k_off + k_off_j
                k = tl.load(k_ptr, mask=tl.arange(BLOCK_N) < BLOCK_N, other=0.0)

                qk = qk + q * k
                m_i = tl.maximum(m_i, qk)
                qk = qk - m_i
                exp_qk = tl.exp(qk)
                acc = acc + exp_qk * V + l_i
                l_i = l_i + exp_qk

        acc = acc / l_i
        o_off_i = i * stride_ok
        o_ptr = o + o_off + o_off_i
        tl.store(o_ptr, acc, mask=tl.arange(BLOCK_M) < BLOCK_M)

# Launch the Triton kernel
def forward(Q, K, V, o, q_scale, k_scale):
    grid = (triton.cdiv(Q.shape[0], BLOCK_M) * triton.cdiv(Q.shape[1], BLOCK_N),)
    _attn_fwd[grid](Q, K, V, o, q_scale, k_scale, Q.stride(0), Q.stride(1), Q.stride(2), K.stride(0), K.stride(1), K.stride(2), V.stride(0), V.stride(1), V.stride(2), o.stride(0), o.stride(1), o.stride(2), Q.shape[0
