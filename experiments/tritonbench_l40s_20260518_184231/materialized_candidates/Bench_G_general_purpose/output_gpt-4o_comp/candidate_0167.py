import triton
import triton.language as tl
import torch

# Define constants for block sizes
BLOCK_M = 128  # Size of block for queries
BLOCK_N = 128  # Size of block for keys and values

@triton.jit
def _attn_fwd_inner(Q, K, V, Q_scale, K_scale, Out, stride_qz, stride_qh, stride_qm, stride_qk,
                    stride_kz, stride_kh, stride_kn, stride_kk,
                    stride_vz, stride_vh, stride_vk, stride_vn,
                    stride_oz, stride_oh, stride_om, stride_on,
                    N_CTX, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Block indices
    m = tl.program_id(0)
    n = tl.program_id(1)

    # Offsets for Q, K, V
    q_offset = m * BLOCK_M
    k_offset = n * BLOCK_N

    # Load blocks of K and V
    K_ptrs = K + k_offset * stride_kn
    V_ptrs = V + k_offset * stride_vk

    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)  # For normalization
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)  # For max score

    # Iterate over context
    for i in range(0, N_CTX, BLOCK_N):
        # Load a block of K
        k = tl.load(K_ptrs + i * stride_kn)
        # Load a block of V
        v = tl.load(V_ptrs + i * stride_vk)

        # Compute scaled dot-product qk
        qk = tl.dot(Q, k) * Q_scale * K_scale

        # Update max score for numerical stability
        m_i = tl.maximum(m_i, tl.max(qk, axis=1))

        # Compute softmax
        p = tl.exp(qk - m_i[:, None])
        l_i += tl.sum(p, axis=1)

        # Accumulate result
        acc += tl.dot(p, v)

    # Normalize by l_i
    acc /= l_i[:, None]

    # Store result in output
    Out_ptrs = Out + q_offset * stride_om + k_offset * stride_on
    tl.store(Out_ptrs, acc)

@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out, stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_oz, stride_oh, stride_om, stride_on,
              N_CTX, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Grid size
    grid = (triton.cdiv(Q.shape[2], BLOCK_M), triton.cdiv(K.shape[2], BLOCK_N))

    # Launch kernel
    _attn_fwd_inner[grid](Q, K, V, Q_scale, K_scale, Out, stride_qz, stride_qh, stride_qm, stride_qk,
                          stride_kz, stride_kh, stride_kn, stride_kk,
                          stride_vz, stride_vh, stride_vk, stride_vn,
                          stride_oz, stride_oh, stride_om, stride_on,
                          N_CTX, BLOCK_M, BLOCK_N)

def scaled_dot_product_attention(Q, K, V, Q_scale, K_scale, stride_qz, stride_qh, stride_qm, stride_qk,
                                 stride_kz, stride_kh, stride_kn, stride_kk,
                                 stride_vz, stride_vh, stride_vk, stride_vn,
                                 stride_oz, stride_oh, stride_om, stride_on):
    # Allocate output tensor
    Out = torch.empty((Q.shape[0], Q.shape[1], Q.shape[2], V.shape[3]), device=Q.device, dtype=Q.dtype)

    # Launch the Triton kernel
    _attn_fwd(Q, K, V, Q_scale, K_scale, Out, stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vk, stride_vn,
              stride_oz, stride_oh, stride_om, stride_on,
              N_CTX=K.shape[2], BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    return Out
