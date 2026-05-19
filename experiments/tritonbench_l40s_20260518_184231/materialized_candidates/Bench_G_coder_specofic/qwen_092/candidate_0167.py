import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    Q_ptr, K_ptr, V_ptr, K_scale_ptr, q_scale, k_scale, Out_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Initialize pointers and offsets
    pid = tl.program_id(axis=0)
    pid_qh = pid
    pid_kh = pid
    pid_oh = pid
    pid_om = pid

    qh = pid_qh
    kh = pid_kh
    oh = pid_oh
    om = pid_om

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K and V blocks
    for k in range(0, N_CTX, BLOCK_K):
        # Load blocks of K and V
        k_base = k * stride_kk
        k_base_q = k_base + pid_kh * stride_kh
        k_base_k = k_base + pid_kh * stride_kh
        k_base_v = k_base + pid_kh * stride_kh

        K_block = tl.load(K_ptr + k_base_k, mask=tl.arange(BLOCK_K) < N_CTX - k, other=0.0)
        V_block = tl.load(V_ptr + k_base_v, mask=tl.arange(BLOCK_K) < N_CTX - k, other=0.0)

        # Scale K block
        K_block_scaled = K_block * tl.load(K_scale_ptr + pid_kh)

        # Compute scaled dot-product qk
        qk = tl.dot(Q_block, K_block_scaled)

        # Apply softmax over qk
        qk -= tl.max(qk, axis=1, keepdims=True)
        qk_exp = tl.exp(qk)
        qk_sum = tl.sum(qk_exp, axis=1, keepdims=True)
        p = qk_exp / qk_sum

        # Accumulate result of p weighted by V into acc
        acc += p * V_block

    # Store the result in Out
    out_base = (qh * stride_qh + om * stride_om) * stride_on
    tl.store(Out_ptr + out_base, acc, mask=tl.arange(BLOCK_M) < BLOCK_M, other=0.0)

@triton.jit
def _attn_fwd(
    Q, K, V, Q_scale, K_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Get grid size
    N_CTX = Q.shape[2]
    grid_size = (N_CTX + BLOCK_M - 1) // BLOCK_M

    # Launch kernel
    for pid in range(grid_size):
        _attn_fwd_inner(
            Q, K, V, K_scale, Q_scale, K_scale, Out,
            stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk,
            stride_vz, stride_vh, stride_vk, stride_vn,
            stride_oz, stride_oh, stride_om, stride_on,
            BLOCK_M, BLOCK_N, BLOCK_K
        )
