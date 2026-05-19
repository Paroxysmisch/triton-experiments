import triton
import triton.language as tl
import torch

# Define constants
H = 12  # Number of attention heads
BLOCK_DMODEL = 64  # Block size for model dimension
BLOCK_M = 128  # Block size for queries
BLOCK_N = 128  # Block size for keys/values

@triton.jit
def _fwd_kernel_int8kv(Q, K, V, Out, softmax_scale, L, M, stride_qz, stride_qh, stride_qm, stride_qd,
                       stride_kz, stride_kh, stride_kn, stride_kd, stride_vz, stride_vh, stride_vn, stride_vd,
                       stride_oz, stride_oh, stride_om, stride_od, Z, N_CTX, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Program IDs
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Q
    q_ptrs = Q + pid_z * stride_qz + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX)

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over K and V
    for n in range(0, N_CTX, BLOCK_N):
        # Load K
        k_ptrs = K + pid_z * stride_kz + pid_h * stride_kh + (n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(n + offs_n)[:, None] < N_CTX)

        # Compute dot product between Q and K
        qk = tl.dot(q, k, trans_b=True)

        # Apply causal mask
        causal_mask = offs_m[:, None] >= (n + offs_n)
        qk = tl.where(causal_mask, qk, float('-inf'))

        # Scale and compute softmax
        qk = qk * softmax_scale
        m_curr = tl.max(qk, axis=1)
        qk = qk - m_curr[:, None]
        p = tl.exp(qk)
        l_curr = tl.sum(p, axis=1)
        p = p / l_curr[:, None]

        # Load V
        v_ptrs = V + pid_z * stride_vz + pid_h * stride_vh + (n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(n + offs_n)[:, None] < N_CTX)

        # Update accumulators
        acc += tl.dot(p, v)

    # Store output
    out_ptrs = Out + pid_z * stride_oz + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc)

def context_attention_fwd_ppl_int8kv(Q, K, V, softmax_scale, L, M, Z, N_CTX):
    # Configure the grid
    grid = (Z, H, (N_CTX + BLOCK_M - 1) // BLOCK_M)

    # Define strides
    stride_qz, stride_qh, stride_qm, stride_qd = Q.stride()
    stride_kz, stride_kh, stride_kn, stride_kd = K.stride()
    stride_vz, stride_vh, stride_vn, stride_vd = V.stride()
    stride_oz, stride_oh, stride_om, stride_od = Q.stride()  # Assuming Out has the same shape as Q

    # Allocate output
    Out = torch.empty_like(Q, dtype=torch.float32)

    # Launch the kernel
    _fwd_kernel_int8kv[grid](
        Q, K, V, Out, softmax_scale, L, M,
        stride_qz, stride_qh, stride_qm, stride_qd,
        stride_kz, stride_kh, stride_kn, stride_kd,
        stride_vz, stride_vh, stride_vn, stride_vd,
        stride_oz, stride_oh, stride_om, stride_od,
        Z, N_CTX, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )

    return Out
