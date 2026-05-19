import triton
import triton.language as tl

# Define constants for block sizes
BLOCK_M = 128
BLOCK_N = 128

@triton.jit
def _attn_fwd_inner(Q, K, V, o, q_scale, k_scale, stride_qm, stride_kn, stride_vn, stride_om, stride_oq, stride_ok, stride_ov, stride_oqk, stride_ovk, M, N, num_heads, head_dim):
    # Calculate block index
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate the start index for each block
    q_start = pid_m * BLOCK_M
    k_start = pid_n * BLOCK_N

    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Iterate over the keys and values
    for k in range(0, N, BLOCK_N):
        # Load blocks of Q, K, V
        q = tl.load(Q + q_start * stride_qm + k * stride_oqk, mask=q_start + tl.arange(0, BLOCK_M) < M)
        k = tl.load(K + k_start * stride_kn + k * stride_ok, mask=k_start + tl.arange(0, BLOCK_N) < N)
        v = tl.load(V + k_start * stride_vn + k * stride_ovk, mask=k_start + tl.arange(0, BLOCK_N) < N)

        # Compute scaled dot product
        qk = tl.dot(q, k, trans_b=True) * q_scale * k_scale

        # Update m_i for numerical stability
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_qk = tl.exp(qk - m_i_new[:, None])

        # Update l_i
        l_i_new = tl.exp(m_i - m_i_new) * l_i + tl.sum(exp_qk, axis=1)

        # Update acc
        acc = (acc * (l_i[:, None] / l_i_new[:, None])) + tl.dot(exp_qk, v)

        # Update m_i and l_i
        m_i = m_i_new
        l_i = l_i_new

    # Normalize the accumulated result
    o_result = acc / l_i[:, None]

    # Write back to the output tensor
    tl.store(o + q_start * stride_om + k_start * stride_ov, o_result)

@triton.jit
def _attn_fwd(Q, K, V, o, q_scale, k_scale, M, N, num_heads, head_dim, stride_qm, stride_kn, stride_vn, stride_om, stride_oq, stride_ok, stride_ov, stride_oqk, stride_ovk):
    # Launch the inner kernel
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _attn_fwd_inner[grid](Q, K, V, o, q_scale, k_scale, stride_qm, stride_kn, stride_vn, stride_om, stride_oq, stride_ok, stride_ov, stride_oqk, stride_ovk, M, N, num_heads, head_dim)

def forward(q, k, v, q_scale, k_scale):
    # Extract dimensions
    M, num_heads, head_dim = q.shape
    N = k.shape[1]

    # Allocate output tensor
    o = torch.empty((M, num_heads, head_dim), device=q.device, dtype=q.dtype)

    # Calculate strides
    stride_qm, stride_oqk = q.stride(0), q.stride(2)
    stride_kn, stride_ok = k.stride(1), k.stride(2)
    stride_vn, stride_ovk = v.stride(1), v.stride(2)
    stride_om, stride_oq, stride_ov = o.stride(0), o.stride(1), o.stride(2)

    # Launch the Triton kernel
    _attn_fwd[(M // BLOCK_M, N // BLOCK_N)](q, k, v, o, q_scale, k_scale, M, N, num_heads, head_dim, stride_qm, stride_kn, stride_vn, stride_om, stride_oq, stride_ok, stride_ov, stride_oqk, stride_ovk)

    return o
