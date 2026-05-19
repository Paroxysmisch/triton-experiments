import math
import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out, Lse, TMP,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    nheads, seqlen_q, seqlen_k, headdim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    
    # Compute batch and head indices
    batch_idx = bid // nheads
    head_idx = bid % nheads
    
    # Initialize row offsets
    row_start = pid * BLOCK_M
    row_offs = row_start + tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)
    
    # Initialize pointers
    q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + row_offs[:, None] * stride_qm
    k_ptrs = K + batch_idx * stride_kb + head_idx * stride_kh
    v_ptrs = V + batch_idx * stride_vb + head_idx * stride_vh
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    
    # Load Q block
    q = tl.load(q_ptrs, mask=row_offs[:, None] < seqlen_q)
    
    # Loop over K,V blocks
    for block_start_k in range(0, seqlen_k, BLOCK_N):
        block_end_k = min(block_start_k + BLOCK_N, seqlen_k)
        
        # Load K,V blocks
        k = tl.load(k_ptrs + block_start_k * stride_kn)
        v = tl.load(v_ptrs + block_start_k * stride_vn)
        
        # Compute attention scores
        qk = tl.dot(q, k, trans_b=True)
        qk = qk * softmax_scale
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            causal_mask = row_offs[:, None] >= (block_start_k + col_offs[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))
        
        # Compute softmax
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Update accumulators
        acc_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_scale[:, None]
        acc_o += tl.dot(p.to(v.dtype), v)
        
        # Update running max/sum
        m_i = tl.maximum(m_i, m_ij)
        l_i = l_i * acc_scale + l_ij
    
    # Final rescaling
    acc_o = acc_o / l_i[:, None]
    
    # Store outputs
    out_ptrs = Out + batch_idx * stride_ob + head_idx * stride_oh + row_offs[:, None] * stride_om
    tl.store(out_ptrs, acc_o, mask=row_offs[:, None] < seqlen_q)
    
    # Store softmax stats
    lse_ptrs = Lse + bid * seqlen_q + row_offs
    tl.store(lse_ptrs, m_i + tl.log(l_i), mask=row_offs < seqlen_q)

def flash_attn_forward(q, k, v, causal=False, softmax_scale=None):
    """
    Forward pass for Flash Attention.
    
    Args:
        q: Query tensor of shape [batch_size, nheads, seqlen_q, headdim]
        k: Key tensor of shape [batch_size, nheads, seqlen_k, headdim]
        v: Value tensor of shape [batch_size, nheads, seqlen_k, headdim]
        causal: Whether to apply causal masking
        softmax_scale: Scaling factor for attention scores (default: 1/sqrt(headdim))
    
    Returns:
        output: Output tensor of shape [batch_size, nheads, seqlen_q, headdim]
        lse: Log-sum-exp values for backward pass
    """
    batch_size, nheads, seqlen_q, headdim = q.shape
    _, _, seqlen_k, _ = k.shape
    
    # Initialize scaling factor
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)
    
    # Allocate output tensors
    output = torch.empty_like(q)
    lse = torch.empty((batch_size, nheads, seqlen_q), device=q.device, dtype=torch.float32)
    tmp = torch.empty_like(lse)
    
    # Configure kernel parameters
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)
    
    # Launch kernel
    grid = (triton.cdiv(seqlen_q, BLOCK_M), batch_size * nheads)
    _fwd_kernel[grid](
        q, k, v, output, lse, tmp,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        nheads, seqlen_q, seqlen_k, headdim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        IS_CAUSAL=causal,
        num_warps=8,
        num_stages=1
    )
    
    return output, lse
