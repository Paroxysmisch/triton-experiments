import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, m_i, l_i,
    q,  # shape: (BLOCK_M, D_HEAD)
    k, v,  # shapes: (BLOCK_N, D_HEAD), (BLOCK_N, D_HEAD)
    q_scale, k_scale,
    seq_len,
    offs_m, offs_n,
    STAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    # Compute QK^T
    s = tl.dot(q, tl.trans(k))  # (BLOCK_M, BLOCK_N)
    s = s.to(tl.float32) * q_scale * k_scale
    s = s * tl.math.rsqrt(tl.math.sqrt(D_HEAD))  # Scaling factor

    if STAGE == 1:
        # Compute local max and sum for softmax
        m_curr = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_curr)
        l_curr = alpha * l_i + tl.sum(tl.exp(s - m_curr[:, None]), axis=1)
        return acc, m_curr, l_curr
    elif STAGE == 2:
        # Compute attention weights and accumulate
        p = tl.exp(s - m_i[:, None]) / l_i[:, None]
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        return acc, m_i, l_i
    else:
        return acc, m_i, l_i

@triton.jit
def _attn_fwd(
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_dim_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_dim_stride,
    v_batch_stride, v_head_stride, v_seq_stride, v_dim_stride,
    o_batch_stride, o_head_stride, o_seq_stride, o_dim_stride,
    q_scale, k_scale,
    seq_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    m_idx = tl.program_id(2)
    
    # Offsets for Q block
    m_start = m_idx * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    
    # Load Q
    q_ptrs = q_ptr + batch_idx * q_batch_stride + head_idx * q_head_stride + \
             (offs_m[:, None] * q_seq_stride + offs_d[None, :] * q_dim_stride)
    mask_q = (offs_m < seq_len)[:, None]
    q = tl.load(q_ptrs, mask=mask_q, other=0.0)
    
    # Initialize accumulator and softmax variables
    acc = tl.zeros((BLOCK_M, D_HEAD), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # Loop over K/V blocks
    num_n_blocks = tl.cdiv(seq_len, BLOCK_N)
    for n_idx in range(num_n_blocks):
        n_start = n_idx * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        
        # Load K
        k_ptrs = k_ptr + batch_idx * k_batch_stride + head_idx * k_head_stride + \
                 (offs_n[None, :] * k_seq_stride + offs_d[:, None] * k_dim_stride)
        mask_k = (offs_n < seq_len)[None, :]
        k = tl.load(k_ptrs, mask=mask_k, other=0.0)
        
        # Load V
        v_ptrs = v_ptr + batch_idx * v_batch_stride + head_idx * v_head_stride + \
                 (offs_n[:, None] * v_seq_stride + offs_d[None, :] * v_dim_stride)
        mask_v = (offs_n < seq_len)[:, None]
        v = tl.load(v_ptrs, mask=mask_v, other=0.0)
        
        # Stage 1: Update softmax stats
        acc, m_i, l_i = _attn_fwd_inner(
            acc, m_i, l_i, q, k, v, q_scale, k_scale, seq_len,
            offs_m, offs_n, 1, BLOCK_M, BLOCK_N, D_HEAD
        )
        
        # Stage 2: Accumulate attention output
        acc, m_i, l_i = _attn_fwd_inner(
            acc, m_i, l_i, q, k, v, q_scale, k_scale, seq_len,
            offs_m, offs_n, 2, BLOCK_M, BLOCK_N, D_HEAD
        )
    
    # Write output
    o_ptrs = o_ptr + batch_idx * o_batch_stride + head_idx * o_head_stride + \
             (offs_m[:, None] * o_seq_stride + offs_d[None, :] * o_dim_stride)
    tl.store(o_ptrs, acc.to(q.dtype), mask=mask_q)

def forward(q, k, v, q_scale, k_scale):
    assert q.dim() == 4, "Q must be 4D (batch, heads, seq_len, dim)"
    batch, heads, seq_len, d_head = q.shape
    
    # Initialize output tensor
    o = torch.empty_like(q)
    
    # Configure kernel grid and launch parameters
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (batch, heads, triton.cdiv(seq_len, BLOCK_M))
    
    # Launch kernel
    _attn_fwd[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q_scale, k_scale,
        seq_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D_HEAD=d_head,
    )
    return o
