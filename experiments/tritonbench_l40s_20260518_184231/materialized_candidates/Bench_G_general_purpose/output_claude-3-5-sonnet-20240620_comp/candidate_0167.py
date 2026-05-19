import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner(
    Q, K, V, sm_scale, Q_scale, K_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Block indices
    num_block_m = tl.cdiv(N_CTX, BLOCK_M)
    num_block_n = tl.cdiv(N_CTX, BLOCK_N)
    
    # Extract batch (z) and head (h) indices
    zh = pid // (num_block_m * num_block_n)
    bhm = pid % (num_block_m * num_block_n)
    
    # Extract z and h
    z = zh // H
    h = zh % H
    
    # Extract block indices
    block_m = bhm // num_block_n
    block_n = bhm % num_block_n
    
    # Compute offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize pointers
    q_ptrs = Q + z * stride_qz + h * stride_qh + block_m * BLOCK_M * stride_qm
    k_ptrs = K + z * stride_kz + h * stride_kh + block_n * BLOCK_N * stride_kn
    v_ptrs = V + z * stride_vz + h * stride_vh + block_n * BLOCK_N * stride_vk
    
    # Load Q block
    q = tl.load(q_ptrs + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    q_scale = tl.load(Q_scale + z * H + h)
    q = q * q_scale
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Iterate over K blocks
    for block_k in range(0, num_block_n):
        # Load K and V blocks
        k = tl.load(k_ptrs + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        v = tl.load(v_ptrs + offs_n[:, None] * stride_vn + offs_d[None, :])
        k_scale = tl.load(K_scale + z * H + h)
        k = k * k_scale
        
        # Compute attention scores
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        
        # Update maximum scores and compute exponentials
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_i_new[:, None])
        l_i_new = l_i * tl.exp(m_i - m_i_new) + tl.sum(p, 1)
        
        # Update accumulators
        acc_scale = tl.exp(m_i - m_i_new) / l_i_new
        acc = acc * acc_scale[:, None]
        p = p / l_i_new[:, None]
        acc += tl.dot(p, v)
        
        # Update running statistics
        l_i = l_i_new
        m_i = m_i_new
        
    # Store output
    out_ptrs = Out + z * stride_oz + h * stride_oh + block_m * BLOCK_M * stride_om
    tl.store(out_ptrs + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on, acc)

@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, Q_scale, K_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    num_block_m = tl.cdiv(N_CTX, BLOCK_M)
    num_block_n = tl.cdiv(N_CTX, BLOCK_N)
    grid = (Z * H * num_block_m * num_block_n,)
    
    _attn_fwd_inner[grid](
        Q, K, V, sm_scale, Q_scale, K_scale,
        Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )

# Python wrapper
def attention_forward(q, k, v, q_scale, k_scale, sm_scale):
    """
    Forward pass for scaled dot-product attention.
    
    Args:
        q: Query tensor of shape (Z, H, M, K)
        k: Key tensor of shape (Z, H, N, K)
        v: Value tensor of shape (Z, H, N, K)
        q_scale: Query scaling factor of shape (Z, H)
        k_scale: Key scaling factor of shape (Z, H)
        sm_scale: Softmax scaling factor (float)
    
    Returns:
        out: Output tensor of shape (Z, H, M, K)
    """
    Z, H, M, K = q.shape
    N = k.shape[2]
    
    # Allocate output
    out = torch.empty_like(q)
    
    # Configure block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = K
    
    # Launch kernel
    _attn_fwd(
        q, k, v, sm_scale, q_scale, k_scale,
        out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return out
