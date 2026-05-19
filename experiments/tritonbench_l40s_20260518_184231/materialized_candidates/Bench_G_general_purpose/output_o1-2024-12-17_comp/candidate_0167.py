import triton
import triton.language as tl

# ------------------------------------------------------------
# Inner kernel: Computes partial contributions to the attention
# ------------------------------------------------------------
@triton.jit
def _attn_fwd_inner(
    Q_ptr, K_ptr, V_ptr,
    Q_scale_ptr, K_scale_ptr,
    Out_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, M, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Program IDs for block-row of Q (m-dim) and block-col of K/V (n-dim)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Batch dimension ID
    pid_z = tl.program_id(2)
    # Head dimension ID
    pid_h = tl.program_id(3)

    # Offsets for each dimension
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Limit checks
    mask_m = offs_m < M
    mask_n = offs_n < N_CTX

    # Pointers for Q (block of rows)
    q_ptrs = Q_ptr + \
             pid_z * stride_qz + \
             pid_h * stride_qh + \
             (offs_m[:, None] * stride_qm) + \
             (tl.arange(0, 1) * stride_qk)
    
    # Pointers for scaling
    q_scale_ptr = Q_scale_ptr + (pid_z * H + pid_h)
    k_scale_ptr = K_scale_ptr + (pid_z * H + pid_h)
    
    # Load Q scale and K scale
    q_scale = tl.load(q_scale_ptr)
    k_scale = tl.load(k_scale_ptr)

    # Initialize partial accumulators for the output
    acc = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # We iterate over the K/V dimension in blocks
    # for a softmax across the entire n-range (context size).
    # Each block accumulates partial sums to maintain numerical stability.
    q_vec = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    # Scale Q
    q_vec = q_vec * q_scale

    # Temporary buffer for partial output
    acc_out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load K block
    k_ptrs = K_ptr + \
             pid_z * stride_kz + \
             pid_h * stride_kh + \
             (offs_n[None, :] * stride_kn) + \
             (tl.arange(0, 1) * stride_kk)
    k_vec = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
    # Scale K
    k_vec = k_vec * k_scale

    # Compute qk = Q * K^T (element-wise for single-embedding step)
    qk = q_vec * tl.transpose(k_vec, 0, 1)
    qk_sum = tl.sum(qk, 1)
    # Numerically stable softmax update
    m_i_new = tl.maximum(m_i, qk_sum)
    alpha = tl.exp(m_i - m_i_new)
    p = tl.exp(qk_sum - m_i_new)
    l_i = alpha * l_i + p
    acc = alpha * acc + p * qk_sum
    m_i = m_i_new

    # Probability matrix for this block
    p_matrix = tl.exp(qk_sum - m_i[:, None])
    p_matrix = p_matrix / tl.sum(p_matrix, 1)[:, None]
    # Save p in acc_out for later multiplication with V
    indices_m = tl.reshape(tl.arange(0, BLOCK_M), (BLOCK_M, 1))
    indices_n = tl.reshape(tl.arange(0, BLOCK_N), (1, BLOCK_N))
    acc_out = tl.where((mask_m[:, None] & mask_n[None, :]),
                       p_matrix,
                       acc_out)

    # Multiply with V
    v_ptrs = V_ptr + \
             pid_z * stride_vz + \
             pid_h * stride_vh + \
             (offs_n[None, :] * stride_vk) + \
             (tl.arange(0, 1) * stride_vn)
    v_val = tl.load(v_ptrs, mask=mask_n[None, :], other=0.0)
    # Weighted sum over block
    weighted_val = acc_out @ v_val

    # Store partial results into Out
    out_ptrs = Out_ptr + \
               pid_z * stride_oz + \
               pid_h * stride_oh + \
               offs_m * stride_om
    tl.store(out_ptrs, weighted_val, mask=mask_m)


# ---------------------------------------------------------
# Outer kernel: Iterates over context blocks, calls _attn_fwd_inner
# ---------------------------------------------------------
@triton.jit
def _attn_fwd(
    Q_ptr, K_ptr, V_ptr,
    Q_scale_ptr, K_scale_ptr,
    Out_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, M, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # We launch a grid of blocks that covers M (query length) and N_CTX (key length), plus batch dims
    # This kernel entry point uses the same program_id layout as _attn_fwd_inner
    _attn_fwd_inner[()](
        Q_ptr, K_ptr, V_ptr,
        Q_scale_ptr, K_scale_ptr,
        Out_ptr,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        Z, H, M, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )


# ---------------------------------------------------------
# Python wrapper to call the Triton kernels
# ---------------------------------------------------------
def attention_fwd(Q, K, V, Q_scale, K_scale, Out,
                  stride_qz, stride_qh, stride_qm, stride_qk,
                  stride_kz, stride_kh, stride_kn, stride_kk,
                  stride_vz, stride_vh, stride_vk, stride_vn,
                  stride_oz, stride_oh, stride_om, stride_on,
                  Z, H, M, N_CTX,
                  BLOCK_M=64, BLOCK_N=64):
    grid = ( (M + BLOCK_M - 1) // BLOCK_M,
             (N_CTX + BLOCK_N - 1) // BLOCK_N,
             Z,
             H )
    _attn_fwd[grid](
        Q, K, V,
        Q_scale, K_scale,
        Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        Z, H, M, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
