import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs, start_m, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, 
    offs_m, offs_n, N_CTX
):
    # Load q: [BLOCK_M, HEAD_DIM]
    q = tl.load(q + offs_m[:, None] * BLOCK_M + offs_n[None, :] * HEAD_DIM)
    q = q * q_scale

    if STAGE == 1:
        # Load k: [BLOCK_N, HEAD_DIM]
        k = tl.load(K_ptrs + offs_n[:, None] * BLOCK_N + offs_n[None, :] * HEAD_DIM)
        k_scale = tl.load(K_scale_ptr + offs_n)
        k = k * k_scale

        # Compute dot product: [BLOCK_M, BLOCK_N]
        dot = tl.dot(q, k, allow_tf32=True)

        # Causal masking
        m_ij = tl.where(offs_m[:, None] >= (offs_n[None, :] + start_m), 0, float('-inf'))
        dot = dot + m_ij

        # Compute maximum value for numerical stability
        m_i_new = tl.maximum(m_i, tl.max(dot, 1))
        alpha = tl.exp(dot - m_i_new[:, None])
        alpha = alpha * tl.where(offs_m[:, None] >= (offs_n[None, :] + start_m), 1.0, 0.0)

        # Update l_i and m_i
        l_i_new = tl.where(offs_m >= start_m, l_i + tl.sum(alpha, 1), l_i)
        m_i = m_i_new

    else:
        # Load k: [BLOCK_N, HEAD_DIM]
        k = tl.load(K_ptrs + offs_n[:, None] * BLOCK_N + offs_n[None, :] * HEAD_DIM)
        k_scale = tl.load(K_scale_ptr + offs_n)
        k = k * k_scale

        # Compute dot product: [BLOCK_M, BLOCK_N]
        dot = tl.dot(q, k, allow_tf32=True)

        # Causal masking
        m_ij = tl.where(offs_m[:, None] >= (offs_n[None, :] + start_m), 0, float('-inf'))
        dot = dot + m_ij

        # Compute alpha
        alpha = tl.exp(dot - m_i[:, None])
        alpha = alpha * tl.where(offs_m[:, None] >= (offs_n[None, :] + start_m), 1.0, 0.0)

        # Update l_i and m_i
        l_i_new = l_i + tl.sum(alpha, 1)
        m_i = m_i

    # Load v: [BLOCK_N, HEAD_DIM]
    v = tl.load(V_ptrs + offs_n[:, None] * BLOCK_N + offs_n[None, :] * HEAD_DIM)

    # Compute weighted value
    acc += tl.dot(alpha, v, allow_tf32=True)

    return acc, l_i_new, m_i

@triton.jit
def _attn_fwd(
    Q, K, V, Q_scale, K_scale, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, 
    stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om, stride_on, Z, H, 
    N_CTX, HEAD_DIM, BLOCK_M, BLOCK_N, STAGE
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = tl.cdiv(N_CTX, BLOCK_N)
    num_pid_in_block = num_pid_m * num_pid_n
    pid_z = pid // num_pid_in_block
    pid_in_block = pid % num_pid_in_block
    pid_m = pid_in_block // num_pid_n
    pid_n = pid_in_block % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_z = pid_z
    offs_h = tl.arange(0, H)

    q_ptrs = Q + (offs_z * stride_qz + offs_h[:, None] * stride_qh + offs_m[None, :] * stride_qm + offs_n[None, :] * stride_qk)
    k_ptrs = K + (offs_z * stride_kz + offs_h[:, None] * stride_kh + offs_n[None, :] * stride_kn + offs_n[None, :] * stride_kk)
    v_ptrs = V + (offs_z * stride_vz + offs_h[:, None] * stride_vh + offs_n[None, :] * stride_vk + offs_n[None, :] * stride_vn)
    out_ptrs = Out + (offs_z * stride_oz + offs_h[:, None] * stride_oh + offs_m[None, :] * stride_om + offs_n[None, :] * stride_on)

    q_scale = Q_scale[offs_z, :]
    k_scale = K_scale[offs_z, :]

    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)

    for start_m in range(0, N_CTX, BLOCK_M):
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q_ptrs, q_scale, k_ptrs, k_scale, v_ptrs, start_m, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX)

    tl.store(out_ptrs, acc)

import torch

def forward(q, k, v, q_scale, k_scale):
    Z, H, N_CTX, HEAD_DIM = q.shape
    BLOCK_M = 16
    BLOCK_N = 16
    STAGE = 1

    # Allocate output tensor
    out = torch.empty((Z, H, N_CTX, HEAD_DIM), device=q.device, dtype=q.dtype)

    # Determine grid size
    grid = (Z * H * (N_CTX // BLOCK_M) * (N_CTX // BLOCK_N),)

    # Launch kernel
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX, HEAD_DIM, BLOCK_M, BLOCK_N, STAGE
    )

    return out
