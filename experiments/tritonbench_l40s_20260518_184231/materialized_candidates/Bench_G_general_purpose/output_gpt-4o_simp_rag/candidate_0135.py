import triton
import triton.language as tl

@triton.jit
def block_sparse_attention_kernel(
    Q, K, V, Out, 
    csr_row_ptr, csr_col_idx, 
    sm_scale, 
    stride_qbs, stride_qh, 
    stride_kbs, stride_kh, 
    stride_vbs, stride_vh, 
    stride_obs, stride_oh, 
    NUM_D_BLOCKS: tl.constexpr, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_D: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    off_q = (
        (cur_batch * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    
    q = tl.load(Q + off_q, mask=offs_m[:, None] < BLOCK_M, other=0.0)

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    row_start = tl.load(csr_row_ptr + start_m)
    row_end = tl.load(csr_row_ptr + start_m + 1)

    for block_idx in range(row_start, row_end):
        col_idx = tl.load(csr_col_idx + block_idx)
        off_k = col_idx * BLOCK_N * stride_kbs + cur_head * stride_kh + offs_d[:, None]
        off_v = col_idx * BLOCK_N * stride_vbs + cur_head * stride_vh + offs_d[None, :]

        k = tl.load(K + off_k, mask=offs_n[None, :] < BLOCK_N, other=0.0)
        v = tl.load(V + off_v, mask=offs_n[:, None] < BLOCK_N, other=0.0)

        qk = tl.dot(q, k) * sm_scale
        qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float('-inf'))

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        p_scale = beta / l_i_new
        p = p * p_scale[:, None]

        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]

        p = p.to(v.dtype)
        acc += tl.dot(p, v)

        l_i = l_i_new
        m_i = m_i_new

    off_o = (
        (cur_batch * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    tl.store(Out + off_o, acc, mask=offs_m[:, None] < BLOCK_M)


def block_sparse_attention_fwd(Q, K, V, Out, csr_row_ptr, csr_col_idx, sm_scale, BLOCK_M, BLOCK_N, BLOCK_D):
    batch_size, num_heads, seq_len, d_model = Q.shape
    grid = (batch_size, num_heads, seq_len // BLOCK_M)

    block_sparse_attention_kernel[grid](
        Q, K, V, Out,
        csr_row_ptr, csr_col_idx,
        sm_scale,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        Out.stride(0), Out.stride(1),
        NUM_D_BLOCKS=1,  # Adjust this if needed
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D
    )
