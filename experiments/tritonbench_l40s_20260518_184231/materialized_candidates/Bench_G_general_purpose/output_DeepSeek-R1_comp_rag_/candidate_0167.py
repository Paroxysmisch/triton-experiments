import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_ptrs, V_ptrs,
    offs_n, offs_d,
    curr_batch_seq_len, curr_batch_start_idx,
    stride_kbs, stride_vbs,
    k_scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Load K and V blocks, compute scaled dot product, update accumulators
    start_n = tl.multiple_of(0, BLOCK_N)
    k = tl.load(K_ptrs + (curr_batch_start_idx + start_n) * stride_kbs, 
                mask=(start_n + offs_n[:, None]) < curr_batch_seq_len, other=0.0) * k_scale
    qk = tl.dot(q, k)
    
    # Apply causal masking
    offs_m = start_n + offs_n
    causal_mask = (offs_m[:, None] <= offs_n[None, :])
    qk = qk * causal_mask + (1 - causal_mask) * float('-inf')
    
    # Compute softmax
    m_ij = tl.max(qk, axis=1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, axis=1)
    
    # Update m_i and l_i
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    
    # Scale p and update acc
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = alpha * l_i / l_i_new
    acc = acc * acc_scale[:, None]
    
    # Load V and accumulate
    v = tl.load(V_ptrs + (curr_batch_start_idx + start_n) * stride_vbs, 
                mask=(start_n + offs_n[:, None]) < curr_batch_seq_len, other=0.0)
    acc += tl.dot(p, v)
    
    return acc, l_i_new, m_i_new

@triton.jit
def _attn_fwd(
    Q, K, V, Q_scale, K_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_qsz, stride_qsh,  # Strides for Q_scale
    stride_ksz, stride_ksh,  # Strides for K_scale
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # 3D grid: batch, head, sequence blocks
    curr_batch = tl.program_id(0)
    curr_head = tl.program_id(1)
    start_m = tl.program_id(2)
    
    # Load scaling factors
    q_scale = tl.load(Q_scale + curr_batch * stride_qsz + curr_head * stride_qsh)
    k_scale = tl.load(K_scale + curr_batch * stride_ksz + curr_head * stride_ksh)
    
    # Offset calculations for Q, K, V
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Q pointer
    Q_ptr = Q + curr_batch * stride_qz + curr_head * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(Q_ptr, mask=offs_m[:, None] < N_CTX, other=0.0) * q_scale
    
    # Initialize accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # K and V pointers
    K_ptr = K + curr_batch * stride_kz + (curr_head % (K.shape[1] // Q.shape[1])) * stride_kh + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    V_ptr = V + curr_batch * stride_vz + (curr_head % (V.shape[1] // Q.shape[1])) * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    
    # Iterate over K/V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_ptr, V_ptr,
            offs_n, offs_d, N_CTX, curr_batch * N_CTX,
            stride_kz, stride_vz, k_scale,
            BLOCK_M, BLOCK_N, BLOCK_DMODEL
        )
    
    # Store output
    Out_ptr = Out + curr_batch * stride_oz + curr_head * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(Out_ptr, acc, mask=offs_m[:, None] < N_CTX)

def context_attention_fwd(q, k, v, q_scale, k_scale, o):
    # Configuration based on hardware and input dimensions
    d_model = q.size(-1)
    assert d_model in [16, 32, 64, 128, 256], "Unsupported feature dimension"
    BLOCK = 128 if torch.cuda.get_device_capability()[0] >= 8 else 64
    
    # Grid and kernel parameters
    grid = (q.size(0), q.size(1), triton.cdiv(q.size(2), BLOCK))
    num_warps = 4 if d_model <= 64 else 8
    
    # Strides for Q_scale and K_scale (assuming shape [batch, heads])
    stride_qsz, stride_qsh = q_scale.stride()
    stride_ksz, stride_ksh = k_scale.stride()
    
    # Launch kernel
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        stride_qsz, stride_qsh, stride_ksz, stride_ksh,
        q.size(2),  # N_CTX
        BLOCK, BLOCK, d_model,
        num_warps=num_warps
    )
