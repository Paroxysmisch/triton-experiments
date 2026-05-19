import torch
import triton
import triton.language as tl

# Define block sizes for efficient memory access
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32  # For key dimension

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i,
    q, k, v,
    q_scale, k_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Block dimensions
    num_m_blocks = tl.cdiv(N, BLOCK_M)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    
    # Block ID
    block_m = pid // num_n_blocks
    block_n = pid % num_n_blocks
    
    # Offsets
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Load query block
    q_ptrs = q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = k + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    
    # Main loop
    for k in range(0, K, BLOCK_K):
        # Load Q, K blocks
        q_block = tl.load(q_ptrs)
        k_block = tl.load(k_ptrs)
        
        # Apply scales
        q_block = q_block * q_scale
        k_block = k_block * k_scale
        
        # Compute attention scores
        scores = tl.dot(q_block, tl.trans(k_block))
        
        # Update running max
        m_i_new = tl.maximum(m_i, tl.max(scores, 1))
        l_i = l_i * tl.exp(m_i - m_i_new) + tl.sum(tl.exp(scores - m_i_new[:, None]), 1)
        m_i = m_i_new
        
        # Update accumulator
        acc = acc * tl.exp(m_i[:, None] - m_i_new[:, None])
        acc += tl.dot(tl.exp(scores - m_i_new[:, None]), v_block)
    
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(
    q, k, v, o,
    q_scale, k_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    acc, l_i, m_i = _attn_fwd_inner(
        0., 0., float("-inf"),
        q, k, v,
        q_scale, k_scale,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        Z, H, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    # Normalize
    acc = acc / l_i[:, None]
    
    # Write output
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    o_ptrs = o + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, acc)

# Python wrapper
def attention_forward(q, k, v, q_scale=1.0, k_scale=1.0):
    """
    Compute scaled dot-product attention: softmax(Q @ K.T) @ V
    """
    # Shape checks
    batch_size, num_heads, seq_len, d_head = q.shape
    assert k.shape == (batch_size, num_heads, seq_len, d_head)
    assert v.shape == (batch_size, num_heads, seq_len, d_head)
    
    # Output tensor
    o = torch.empty_like(q)
    
    # Grid and block sizes
    grid = (triton.cdiv(seq_len, BLOCK_M) * triton.cdiv(seq_len, BLOCK_N),)
    
    # Launch kernel
    _attn_fwd[grid](
        q, k, v, o,
        q_scale, k_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch_size, num_heads, seq_len, d_head,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return o
