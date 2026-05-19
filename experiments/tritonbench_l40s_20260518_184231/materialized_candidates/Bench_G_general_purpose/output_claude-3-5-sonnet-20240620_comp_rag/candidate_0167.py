import triton
import triton.language as tl
import torch

# Cache for JIT-compiled kernel
cached_kernel = None

@triton.jit
def _attn_fwd_inner(
    Q, K, V, Q_scale, K_scale,
    K_ptrs, K_scale_ptr, V_ptrs,
    acc, l_i, m_i,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    # Load Q block
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Compute attention scores
    q_scale = tl.load(Q_scale)
    k_scale = tl.load(K_scale_ptr)
    scale = q_scale * k_scale
    
    # Load K block and compute QK^T
    k = tl.load(K_ptrs)
    qk = tl.dot(Q, k) * scale
    
    # Compute softmax
    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    
    # Update running max/sum
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    
    # Scale p and acc
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc = acc * acc_scale[:, None]
    
    # Load V block and compute attention
    v = tl.load(V_ptrs)
    acc += tl.dot(p.to(v.dtype), v)
    
    return acc, l_i_new, m_i_new

@triton.jit
def _attn_fwd(
    Q, K, V, Q_scale, K_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    # Program ID
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Initialize pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize accumulator and running max/sum
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Main loop
    for start_n in range(0, N_CTX, BLOCK_N):
        # Compute K/V pointers for current block
        k_ptrs = K + (pid_z * stride_kz + pid_h * stride_kh + 
                     start_n * stride_kn + offs_d[:, None] * stride_kk)
        v_ptrs = V + (pid_z * stride_vz + pid_h * stride_vh + 
                     start_n * stride_vk + offs_d[None, :] * stride_vn)
        
        # Update accumulator
        acc, l_i, m_i = _attn_fwd_inner(
            Q, K, V, Q_scale, K_scale,
            k_ptrs, K_scale + pid_h, v_ptrs,
            acc, l_i, m_i,
            BLOCK_M, BLOCK_N, BLOCK_DMODEL
        )
    
    # Write output
    offs_o = (pid_z * stride_oz + pid_h * stride_oh + 
              offs_m[:, None] * stride_om + offs_d[None, :] * stride_on)
    out_ptrs = Out + offs_o
    tl.store(out_ptrs, acc)

def context_attention_fwd(q, k, v, q_scale, k_scale):
    """
    Wrapper function for attention forward pass
    
    Args:
        q: Query tensor [B, H, M, K]
        k: Key tensor [B, H, N, K] 
        v: Value tensor [B, H, N, K]
        q_scale: Query scaling factor [H]
        k_scale: Key scaling factor [H]
    Returns:
        out: Output tensor [B, H, M, K]
    """
    batch, heads, seq_len, dim = q.shape
    
    # Allocate output
    out = torch.empty_like(q)
    
    # Configure block sizes based on GPU capability
    BLOCK = 128 if torch.cuda.get_device_capability()[0] >= 8 else 64
    
    # Launch kernel
    grid = (batch, heads, triton.cdiv(seq_len, BLOCK))
    
    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch, heads, seq_len,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_DMODEL=dim,
        num_warps=4 if dim <= 64 else 8,
        num_stages=1
    )
    
    return out
