import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    Q, K, V, Q_scale, K_scale, Out,
    stride_qm, stride_qk, stride_kn, stride_kk, stride_vk, stride_vn, stride_om, stride_on,
    m_i, l_i, acc,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    q_scale: tl.constexpr, k_scale: tl.constexpr
):
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for Q, K, V
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load Q, K, V blocks
    Q_block = tl.load(Q + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk)
    K_block = tl.load(K + offs_m[:, None] * stride_kn + offs_n[None, :] * stride_kk)
    V_block = tl.load(V + offs_m[:, None] * stride_vk + offs_n[None, :] * stride_vn)

    # Compute scaled dot-product
    qk = tl.dot(Q_block, K_block, allow_tf32=True) * q_scale * k_scale

    # Compute softmax
    m_i = tl.max(qk, 1)
    qk = qk - m_i[:, None]
    p = tl.exp(qk)
    l_i = tl.sum(p, 1)
    p = p / l_i[:, None]

    # Accumulate result
    acc += tl.dot(p, V_block, allow_tf32=True)

    # Update normalization and maximum score
    l_i = l_i + tl.sum(acc, 1)
    m_i = tl.max(m_i, 0)

    # Store results
    tl.store(Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc)

@triton.jit
def _attn_fwd(
    Q, K, V, Q_scale, K_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    N_CTX: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    q_scale: tl.constexpr, k_scale: tl.constexpr
):
    # Block indices
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Offsets for Q, K, V
    offs_z = pid_z * stride_qz
    offs_h = pid_h * stride_qh
    offs_m = pid_m * BLOCK_M

    # Initialize pointers
    Q_ptr = Q + offs_z + offs_h + offs_m * stride_qm
    K_ptr = K + offs_z + offs_h
    V_ptr = V + offs_z + offs_h
    Out_ptr = Out + offs_z + offs_h + offs_m * stride_om

    # Initialize normalization and maximum score
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over context size
    for start_n in range(0, N_CTX, BLOCK_N):
        _attn_fwd_inner(
            Q_ptr, K_ptr, V_ptr, Q_scale, K_scale, Out_ptr,
            stride_qm, stride_qk, stride_kn, stride_kk, stride_vk, stride_vn, stride_om, stride_on,
            m_i, l_i, acc,
            BLOCK_M, BLOCK_N, q_scale, k_scale
        )
        K_ptr += BLOCK_N * stride_kn
        V_ptr += BLOCK_N * stride_vn

    # Store final results
    tl.store(Out_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_om + tl.arange(0, BLOCK_N)[None, :] * stride_on, acc)

### Wrapper Function

def attn_fwd(Q, K, V, Q_scale, K_scale, Out, N_CTX, BLOCK_M, BLOCK_N, q_scale, k_scale):
    # Grid and block dimensions
    grid = (Q.shape[0], Q.shape[1], (Q.shape[2] + BLOCK_M - 1) // BLOCK_M)
    _attn_fwd[grid](
        Q, K, V, Q_scale, K_scale, Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        N_CTX, BLOCK_M, BLOCK_N, q_scale, k_scale
    )

import torch

# Example input tensors
Q = torch.randn((1, 8, 128, 64), device='cuda')
K = torch.randn((1, 8, 128, 64), device='cuda')
V = torch.randn((1, 8, 128, 64), device='cuda')
Q_scale = torch.tensor(1.0, device='cuda')
K_scale = torch.tensor(1.0, device='cuda')
Out = torch.empty((1, 8, 128, 64), device='cuda')

# Parameters
N_CTX = 128
BLOCK_M = 16
BLOCK_N = 16
q_scale = 1.0
k_scale = 1.0

# Call the wrapper function
attn_fwd(Q, K, V, Q_scale, K_scale, Out, N_CTX, BLOCK_M, BLOCK_N, q_scale, k_scale)
