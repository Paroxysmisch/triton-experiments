import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i,
    q, q_scale,
    K_ptrs, K_scale_ptr,
    V_ptrs,
    start_m, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE,
    offs_m, offs_n, N_CTX):
    # Constants
    N_ITER = N_CTX // BLOCK_N if STAGE == 1 else 1
    IS_STAGE2 = STAGE == 2
    # Load parameters
    q = tl.load(q + offs_m + tl.arange(0, BLOCK_M)[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :])
    q = q * q_scale
    if STAGE == 1:
        m_ij = tl.full([BLOCK_M, BLOCK_N], float('-inf'), dtype=tl.float32)
        l_ij = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        k_scale = tl.load(K_scale_ptr + offs_n + tl.arange(0, BLOCK_N)[None, :])
    for _ in range(N_ITER):
        k = tl.load(K_ptrs + offs_n + tl.arange(0, BLOCK_N)[None, :])
        k = tl.trans(k)
        if STAGE == 1:
            k_mask = tl.where((start_m[:, None] + tl.arange(0, BLOCK_M)[None, :]) <= (start_m[None, :] + tl.arange(0, BLOCK_N)[:, None]), 0., float('-inf'))
            qk = tl.dot(q.to(k.dtype), k) * k_scale[None, :]
            qk += k_mask
        else:
            qk = tl.dot(q.to(k.dtype), k) * k_scale[None, :]
        qk = tl.where(tl.arange(0, BLOCK_M)[:, None] >= tl.arange(0, BLOCK_N)[None, :], qk, float('-inf'))
        if STAGE == 1:
            m_ij_new = tl.maximum(m_ij, tl.max(qk, 1)[:, None])
            l_ij = l_ij * tl.exp(m_ij - m_ij_new) + tl.exp(qk - m_ij_new[:, None])
            m_ij = m_ij_new
        else:
            l_ij = l_ij + tl.exp(qk - m_ij)
        l_i = l_i * tl.exp(m_i - l_i) + l_ij
        m_i = tl.maximum(m_i, tl.max(qk, 1))
        if IS_STAGE2:
            alpha = tl.exp(qk - m_i[:, None])
            acc = acc * tl.exp(m_i - l_i)[:, None] + tl.dot(alpha.to(tl.float16), V_ptrs[offs_n + tl.arange(0, BLOCK_N)[None, :]])
            return acc, l_i, m_i
        else:
            alpha = tl.exp(qk - m_i[:, None])
            acc = acc * tl.exp(m_i - l_i)[:, None] + tl.dot(alpha.to(tl.float16), V_ptrs[offs_n + tl.arange(0, BLOCK_N)[None, :]])
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(
    Q, K, V,
    Q_scale, K_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr):
    # Set up parameters for block processing
    start_m = tl.program_id(0) * BLOCK_M
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    # Compute required indices
    q_index = q_offset + tl.arange(0, BLOCK_M) * stride_qm
    qk_index = q_offset + tl.arange(0, BLOCK_M) * stride_qm + tl.arange(0, BLOCK_N) * stride_qk
    k_index = k_offset + tl.arange(0, BLOCK_N) * stride_kn
    v_index = v_offset + tl.arange(0, BLOCK_N) * stride_vk
    o_index = o_offset + tl.arange(0, BLOCK_M) * stride_om
    # Set up grid for parallel execution
    grid = (tl.cdiv(N_CTX, BLOCK_M), Z * H)
    # Initialize accumulator and m_i/l_i
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + float('-inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Compute attention scores and values
    q = tl.load(Q + q_index)
    q_scale = tl.load(Q_scale + q_index)
    K_ptrs = K + k_index
    V_ptrs = V + v_index
    if STAGE == 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale, V_ptrs, start_m, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, q_index, k_index, N_CTX)
        tl.debug_barrier()
    elif STAGE == 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale, V_ptrs, start_m, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, q_index, k_index, N_CTX)
    tl.store(Out + o_index, acc)

class AttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q_scale, k_scale
