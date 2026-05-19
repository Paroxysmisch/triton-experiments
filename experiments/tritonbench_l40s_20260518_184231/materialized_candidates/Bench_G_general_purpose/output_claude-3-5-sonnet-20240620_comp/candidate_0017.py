import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q, q_scale,
    K_ptrs, K_scale_ptr, V_ptrs,
    start_m, BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Initialize pointers
    offs_k = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Load query block
    q = tl.load(q + offs_k)
    q_scale = tl.load(q_scale)
    q = q * q_scale
    
    # Key block pointer
    k_ptrs = K_ptrs + offs_k[:, None] * BLOCK_N + offs_n[None, :]
    k_scale = tl.load(K_scale_ptr)
    
    if STAGE == 1:
        # Load and process key block
        k = tl.load(k_ptrs)
        k = k * k_scale
        
        # Compute attention scores
        qk = tl.dot(q, k)
        
        # Apply causal mask
        mask = offs_m[:, None] >= (start_m + offs_n[None, :])
        qk = tl.where(mask, qk, float("-inf"))
        
        # Update running maximum
        m_ij = tl.max(qk, 1)
        l_ij = tl.exp(qk - m_ij[:, None])
        l_i += tl.sum(l_ij, 1)
        m_i = tl.maximum(m_i, m_ij)
        
    else:  # STAGE == 2
        # Load value block
        v_ptrs = V_ptrs + offs_k[:, None] * BLOCK_N + offs_n[None, :]
        v = tl.load(v_ptrs)
        
        # Compute softmax and weighted sum
        l_ij = tl.exp(qk - m_i[:, None])
        p = l_ij / l_i[:, None]
        acc += tl.dot(p.to(v.dtype), v)

@triton.jit
def _attn_fwd(
    Q, K, V, Q_scale, K_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute indices
    z = pid // (H * tl.cdiv(N_CTX, BLOCK_M))
    h = (pid % (H * tl.cdiv(N_CTX, BLOCK_M))) // tl.cdiv(N_CTX, BLOCK_M)
    start_m = (pid % tl.cdiv(N_CTX, BLOCK_M)) * BLOCK_M
    
    # Initialize pointers
    q_ptrs = Q + z * stride_qz + h * stride_qh + start_m * stride_qm
    k_ptrs = K + z * stride_kz + h * stride_kh
    v_ptrs = V + z * stride_vz + h * stride_vh
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    
    # Main computation
    for start_n in range(0, N_CTX, BLOCK_N):
        _attn_fwd_inner(
            acc, l_i, m_i, q_ptrs, Q_scale,
            k_ptrs + start_n * stride_kn,
            K_scale + start_n * stride_kk,
            v_ptrs + start_n * stride_vn,
            start_m, BLOCK_M, HEAD_DIM, BLOCK_N,
            STAGE, start_m, start_n, N_CTX
        )
    
    # Store output
    if STAGE == 2:
        out_ptrs = Out + z * stride_oz + h * stride_oh + start_m * stride_om
        tl.store(out_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_on,
                 acc.to(Out.dtype.element_ty))

def forward(q, k, v, q_scale, k_scale):
    """
    Forward pass of blockwise attention.
    
    Args:
        q: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        v: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
        q_scale: Query scaling factor
        k_scale: Key scaling factor
    
    Returns:
        Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Validate input dimensions
    assert k.shape == v.shape == (batch_size, num_heads, seq_len, head_dim)
    
    # Output tensor
    output = torch.empty_like(q)
    
    # Configure grid
    grid = (batch_size * num_heads * triton.cdiv(seq_len, 32),)
    
    # Launch kernel (two stages)
    for stage in [1, 2]:
        _attn_fwd[grid](
            q, k, v, q_scale, k_scale, output,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            batch_size, num_heads, seq_len,
            HEAD_DIM=head_dim,
            BLOCK_M=32,
            BLOCK_N=32,
            STAGE=stage,
        )
    
    return output
