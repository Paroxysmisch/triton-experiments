import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_int8kv(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_qm, stride_kn, stride_vm,
    H, N_CTX, BLOCK_DMODEL, BLOCK_M, BLOCK_N,
):
    # Compute program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = tl.cdiv(N_CTX, BLOCK_N)
    num_pid_in_block = num_pid_m * num_pid_n
    block_id = pid // num_pid_in_block
    pid_mn = pid % num_pid_in_block
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    # Compute block bounds
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (block_id * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    k_ptrs = K + (block_id * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    v_ptrs = V + (block_id * stride_vh + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd)
    o_ptrs = Out + (block_id * stride_oh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_od)

    # Initialize L and M
    L = tl.zeros([BLOCK_M], dtype=tl.float32)
    M = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')

    # Compute QK^T
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_block_ptr = k_ptrs + start_n * stride_kn
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k in range(0, BLOCK_DMODEL, 16):
            q = tl.load(q_ptrs + k)
            k = tl.load(k_block_ptr + k)
            qk += tl.dot(q, k, trans_b=True)
        qk = qk * (1.0 / tl.sqrt(BLOCK_DMODEL))
        qk = tl.where(offs_m[:, None] >= (offs_n[None, :] + start_n), qk, float('-inf'))
        m_i = tl.max(qk, 1)
        l_i = tl.sum(tl.exp(qk - m_i[:, None]), 1)
        m_i = tl.where(m_i > M, m_i, M)
        l_i = tl.where(m_i > M, l_i, l_i * tl.exp(M - m_i))
        L = L + l_i
        M = m_i

    # Compute softmax and out
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_block_ptr = v_ptrs + start_n * stride_vm
        p = tl.load(q_ptrs + start_n * stride_qm)
        p = tl.exp(p - M[:, None])
        p = p / L[:, None]
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        for k in range(0, BLOCK_DMODEL, 16):
            v = tl.load(v_block_ptr + k)
            acc += tl.dot(p, v)
        tl.store(o_ptrs + start_n * stride_qm, acc)

import triton
import triton.language as tl

def context_attention_fwd_ppl_int8kv(Q, K, V, Out, H, N_CTX, BLOCK_DMODEL, BLOCK_M, BLOCK_N):
    # Compute grid size
    grid = (H * (N_CTX // BLOCK_M) * (N_CTX // BLOCK_N),)

    # Set strides
    stride_qb = Q.shape[0] * Q.shape[1] * Q.shape[2]
    stride_qh = Q.shape[1] * Q.shape[2]
    stride_qd = 1
    stride_kb = K.shape[0] * K.shape[1] * K.shape[2]
    stride_kh = K.shape[1] * K.shape[2]
    stride_kd = 1
    stride_vb = V.shape[0] * V.shape[1] * V.shape[2]
    stride_vh = V.shape[1] * V.shape[2]
    stride_vd = 1
    stride_ob = Out.shape[0] * Out.shape[1] * Out.shape[2]
    stride_oh = Out.shape[1] * Out.shape[2]
    stride_od = 1
    stride_qm = Q.shape[2]
    stride_kn = K.shape[2]
    stride_vm = V.shape[2]

    # Adjust block size based on GPU architecture
    if triton.get_device_name() == 'Tesla':
        BLOCK_M = 128
        BLOCK_N = 128

    # Launch kernel
    _fwd_kernel_int8kv[grid](
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kd,
        stride_vb, stride_vh, stride_vd,
        stride_ob, stride_oh, stride_od,
        stride_qm, stride_kn, stride_vm,
        H, N_CTX, BLOCK_DMODEL, BLOCK_M, BLOCK_N,
    )
