import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs, start_m, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX):
    # Load Q vectors
    q_vec = tl.load(q + offs_m * HEAD_DIM)
    q_vec = q_vec * q_scale

    # Initialize accumulators
    acc.fill_(0)
    l_i.fill_(0)
    m_i.fill_(-float('inf'))

    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K and V vectors
        k_vec = tl.load(K_ptrs + start_n * HEAD_DIM)
        k_scale = tl.load(K_scale_ptr + start_n)
        v_vec = tl.load(V_ptrs + start_n * HEAD_DIM)

        # Compute QK^T and apply scaling
        dot_product = tl.dot(q_vec, k_vec)
        scaled_dot_product = dot_product * k_scale

        # Apply causal mask
        if STAGE == 1:
            mask = offs_n < (start_m + BLOCK_M)
            scaled_dot_product = tl.where(mask, scaled_dot_product, -float('inf'))

        # Softmax calculation
        m_ij = tl.maximum(m_i, scaled_dot_product)
        exp_val = tl.exp(scaled_dot_product - m_ij)
        l_i = l_i * tl.exp(m_i - m_ij) + exp_val
        acc = acc * tl.exp(m_i - m_ij) + exp_val * v_vec
        m_i = m_ij

    # Final normalization
    acc /= l_i

@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om, stride_on, Z, H, N_CTX, HEAD_DIM, BLOCK_M, BLOCK_N, STAGE):
    # Compute block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute linear indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Memory pointers
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qk
    k_ptrs = K + offs_m[:, None] * stride_kn + offs_n[None, :] * stride_kk
    v_ptrs = V + offs_m[:, None] * stride_vk + offs_n[None, :] * stride_vn
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)

    # Execute the inner attention computation
    _attn_fwd_inner(acc, l_i, m_i, q_ptrs, Q_scale, k_ptrs, K_scale, v_ptrs, offs_m, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX)

    # Store the result
    tl.store(out_ptrs, acc)

def forward(q, k, v, q_scale, k_scale):
    # Determine dimensions
    Z, H, N_CTX, HEAD_DIM = q.shape

    # Allocate output
    Out = torch.empty((Z, H, N_CTX, HEAD_DIM), device=q.device, dtype=q.dtype)

    # Define block sizes
    BLOCK_M = 64
    BLOCK_N = 64

    # Configure grid
    grid = (triton.cdiv(N_CTX, BLOCK_M), triton.cdiv(N_CTX, BLOCK_N))

    # Launch kernel
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, Out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        Z, H, N_CTX, HEAD_DIM, BLOCK_M, BLOCK_N, STAGE=1
    )

    return Out
