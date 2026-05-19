import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_ptr, V_ptr, K_scale_ptr,  # Modified to include K_scale_ptr
    stride_kn, stride_kk, stride_vk, stride_vn,
    start_n, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    k_scale  # Added k_scale as a parameter
):
    # Calculate offsets for the current block
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    K_ptrs = K_ptr + offs_n[:, None] * stride_kn + tl.arange(0, BLOCK_K)[None, :] * stride_kk
    V_ptrs = V_ptr + offs_n[:, None] * stride_vk + tl.arange(0, BLOCK_K)[None, :] * stride_vn

    # Load K and V, apply scaling
    k = tl.load(K_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
    k = k * k_scale  # Apply K scaling
    v = tl.load(V_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

    # Compute qk
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk = qk.to(tl.float32)  # Cast to float32 for stability

    # Compute current block statistics
    m_ij = tl.max(qk, 1)
    m_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_new)
    beta = tl.exp(m_ij - m_new)

    # Update accumulator and normalization
    p = tl.exp(qk - m_new[:, None])
    l_ij = tl.sum(p, 1) * beta + alpha * l_i
    acc_scale = beta / l_ij
    acc_scale = acc_scale.to(acc.dtype)
    acc = acc * alpha[:, None] + (p.to(v.dtype) @ v) * acc_scale[:, None]

    # Update m_i and l_i
    return acc, l_ij, m_new

@triton.jit
def _attn_fwd(
    Q, K, V, Q_scale, K_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ID management
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Offset calculations for Q
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    Q_ptr = Q + pid_z * stride_qz + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(Q_ptr, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Load scaling factors
    q_scale = tl.load(Q_scale + pid_z * H + pid_h)  # Assume Q_scale is (Z, H)
    k_scale_val = tl.load(K_scale + pid_z * H + pid_h)  # Assume K_scale is (Z, H)
    q = q * q_scale  # Apply Q scaling

    # Initialize accumulation variables
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # Loop over K/V blocks
    K_ptr = K + pid_z * stride_kz + pid_h * stride_kh
    V_ptr = V + pid_z * stride_vz + pid_h * stride_vh
    for start_n in range(0, tl.cdiv(N_CTX, BLOCK_N)):
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            K_ptr, V_ptr, K_scale,
            stride_kn, stride_kk, stride_vk, stride_vn,
            start_n, N_CTX,
            BLOCK_M, BLOCK_N, BLOCK_K,
            k_scale_val  # Pass k_scale as a scalar
        )

    # Write output
    Out_ptr = Out + pid_z * stride_oz + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    tl.store(Out_ptr, acc, mask=offs_m[:, None] < N_CTX)

def attn_forward(q, k, v, q_scale, k_scale, output):
    # Validate inputs
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Inputs must be 4D"
    Z, H, M, K = q.shape
    N_CTX = k.shape[2]

    # Configure kernel grid
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64 if K > 64 else K
    grid = (Z, H, triton.cdiv(M, BLOCK_M))

    # Launch kernel
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return output
