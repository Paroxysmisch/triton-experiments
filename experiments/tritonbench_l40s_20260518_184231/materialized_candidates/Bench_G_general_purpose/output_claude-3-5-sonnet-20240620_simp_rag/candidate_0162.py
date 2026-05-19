import triton
import triton.language as tl

@triton.jit
def _attn_fwd(
    Q, K, V,                     # Query, Key, Value tensors
    Q_scale, K_scale,            # Scaling factors
    Out,                         # Output tensor
    stride_qz, stride_qh,        # Strides for Q
    stride_kz, stride_kh,        # Strides for K
    stride_vz, stride_vh,        # Strides for V
    stride_oz, stride_oh,        # Strides for output
    N_CTX: tl.constexpr,        # Context size
    BLOCK_M: tl.constexpr,      # Block size for M dimension
    BLOCK_N: tl.constexpr,      # Block size for N dimension
    BLOCK_DMODEL: tl.constexpr  # Block size for D dimension
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_z = tl.program_id(2)

    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Initialize pointers
    q_ptrs = Q + pid_z * stride_qz + pid_h * stride_qh + offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :]
    k_ptrs = K + pid_z * stride_kz + pid_h * stride_kh
    v_ptrs = V + pid_z * stride_vz + pid_h * stride_vh

    # Initialize accumulator and scaling
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Load Q block
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    q = q * Q_scale

    # Main loop
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K and V blocks
        k_block_ptrs = k_ptrs + start_n * BLOCK_DMODEL + offs_n[None, :] * BLOCK_DMODEL + offs_d[:, None]
        v_block_ptrs = v_ptrs + start_n * BLOCK_DMODEL + offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :]
        
        k = tl.load(k_block_ptrs, mask=(start_n + offs_n[None, :]) < N_CTX, other=0.0)
        v = tl.load(v_block_ptrs, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        
        k = k * K_scale

        # Compute attention scores
        qk = tl.dot(q, k)
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        
        # Update running statistics
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        
        # Update accumulator
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        
        # Update statistics
        l_i = l_i_new
        m_i = m_i_new

    # Write output
    out_ptrs = Out + pid_z * stride_oz + pid_h * stride_oh + offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :]
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)

def context_attention_forward(q, k, v, scale=None):
    """
    Wrapper function for the forward attention kernel
    
    Args:
        q: Query tensor [batch, heads, seq_len, d_model]
        k: Key tensor [batch, heads, seq_len, d_model]
        v: Value tensor [batch, heads, seq_len, d_model]
        scale: Optional scaling factor
    """
    batch, heads, seq_len, d_model = q.shape
    assert k.shape == v.shape == (batch, heads, seq_len, d_model)
    
    # Determine block sizes based on GPU capability
    BLOCK = 128 if triton.runtime.driver.utils.get_compute_capability()[0] >= 8 else 64
    
    # Initialize output tensor
    output = torch.empty_like(q)
    
    # Calculate scales
    q_scale = k_scale = 1.0 / (d_model ** 0.5) if scale is None else scale
    
    # Launch kernel
    grid = (
        triton.cdiv(seq_len, BLOCK),  # M dimension
        heads,                         # H dimension
        batch                         # Z dimension
    )
    
    _attn_fwd[grid](
        q, k, v,
        q_scale, k_scale,
        output,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        output.stride(0), output.stride(1),
        N_CTX=seq_len,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_DMODEL=d_model,
        num_warps=4 if d_model <= 64 else 8,
        num_stages=1
    )
    
    return output
