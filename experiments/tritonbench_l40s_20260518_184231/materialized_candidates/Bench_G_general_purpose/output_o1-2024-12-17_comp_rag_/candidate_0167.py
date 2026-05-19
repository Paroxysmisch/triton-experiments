import triton
import triton.language as tl

@tl.inline
def _attn_fwd_inner(
    q_block, 
    K, 
    V, 
    Q_scale, 
    K_scale,
    m_i, 
    l_i, 
    acc,
    k_ptrs, 
    v_ptrs,
    start_n,
    cur_seq_len,
    offs_m, 
    offs_n, 
    offs_d,
    stride_kz, 
    stride_kh, 
    stride_kn, 
    stride_kk,
    stride_vz, 
    stride_vh, 
    stride_vk, 
    stride_vn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # load K block
    k = tl.load(
        K + k_ptrs + (start_n + offs_n[None, :]) * stride_kn,
        mask=(start_n + offs_n[None, :]) < cur_seq_len,
        other=0.0
    )
    # compute scaled QK
    qk = tl.dot(q_block, k)
    # apply scaling
    q_scale_val = tl.load(Q_scale)
    k_scale_val = tl.load(K_scale)
    sm_scale = q_scale_val * k_scale_val
    qk = qk * sm_scale

    # causal masking or typical constraints
    qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

    # softmax update
    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)

    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij

    p_scale = beta / l_i_new
    p = p * p_scale[:, None]

    acc_scale = alpha * (l_i / l_i_new)
    acc = acc * acc_scale[:, None]

    # load V block
    v = tl.load(
        V + v_ptrs + (start_n + offs_n[:, None]) * stride_vk,
        mask=(start_n + offs_n[:, None]) < cur_seq_len,
        other=0.0
    )
    p = p.to(v.dtype)
    acc += tl.dot(p, v)

    return m_i_new, l_i_new, acc

@triton.jit
def _attn_fwd(
    Q, 
    K, 
    V, 
    Q_scale, 
    K_scale,
    Out,
    stride_qz, 
    stride_qh, 
    stride_qm, 
    stride_qk,
    stride_kz, 
    stride_kh, 
    stride_kn, 
    stride_kk,
    stride_vz, 
    stride_vh, 
    stride_vk, 
    stride_vn,
    stride_oz, 
    stride_oh, 
    stride_om, 
    stride_on,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # program ids
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    start_m = tl.program_id(2)

    # offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_M)  # For dmodel==BLOCK_M in this example

    # pointer offsets in Q
    q_ptrs = (batch_id * stride_qz) + (head_id * stride_qh)
    # actual pointer referencing each row in Q
    q_block_ptr = Q + (offs_m[:, None] * stride_qm) + (offs_d[None, :] * stride_qk) + q_ptrs

    # load Q block
    q_block = tl.load(
        q_block_ptr,
        mask=offs_m[:, None] < N_CTX,
        other=0.
