import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out, B_Start_Loc, B_Seqlen,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vk, stride_vn,
    stride_ob, stride_oh, stride_om, stride_on,
    window_size, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    batch_id = pid // (num_heads * num_blocks)
    head_id = (pid % (num_heads * num_blocks)) // num_blocks
    block_id = pid % num_blocks

    # Initialize pointers to Q, K, V
    seq_start = B_Start_Loc[batch_id]
    seq_len = B_Seqlen[batch_id]
    
    # Initialize offsets
    offset_q = seq_start + block_id * BLOCK_M
    offset_k = seq_start
    offset_v = seq_start

    # Load Q block
    q_ptrs = Q + offset_q * stride_qm + head_id * stride_qh + batch_id * stride_qb
    q = tl.load(q_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_qm +
                tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk)

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    scale = sm_scale

    # Initialize max scores for numerical stability
    m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Sliding window bounds
    window_start = tl.maximum(0, block_id * BLOCK_M - window_size)
    window_end = tl.minimum(seq_len, (block_id + 1) * BLOCK_M + window_size)

    # Iterate over K,V blocks within window
    for block_start in range(window_start, window_end, BLOCK_N):
        # Load K block
        k_ptrs = K + (offset_k + block_start) * stride_kn + head_id * stride_kh + batch_id * stride_kb
        k = tl.load(k_ptrs + tl.arange(0, BLOCK_N)[:, None] * stride_kn +
                    tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kk)

        # Compute attention scores
        qk = tl.dot(q, tl.trans(k))
        qk = qk * scale

        # Update max scores and compute exponentials
        m_prev = m
        m = tl.maximum(m, tl.max(qk, 1))
        exp_qk = tl.exp(qk - m[:, None])
        
        # Update accumulator
        l_prev = l
        l = l_prev * tl.exp(m_prev - m) + tl.sum(exp_qk, 1)

        # Load V block and update output
        v_ptrs = V + (offset_v + block_start) * stride_vk + head_id * stride_vh + batch_id * stride_vb
        v = tl.load(v_ptrs + tl.arange(0, BLOCK_N)[:, None] * stride_vk +
                    tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vn)
        
        acc = acc * tl.exp(m_prev - m)[:, None] + tl.dot(exp_qk, v)

    # Write output
    out_ptrs = Out + offset_q * stride_om + head_id * stride_oh + batch_id * stride_ob
    out = acc / l[:, None]
    tl.store(out_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_om +
             tl.arange(0, BLOCK_DMODEL)[None, :] * stride_on, out)

def context_attention_fwd(q, k, v, window_size, sm_scale, b_start_loc, b_seqlen):
    """
    Forward pass for sliding window attention.
    
    Args:
        q: Query tensor of shape (batch, heads, seq_len, d_model)
        k: Key tensor of shape (batch, heads, seq_len, d_model)
        v: Value tensor of shape (batch, heads, seq_len, d_model)
        window_size: Size of the attention window
        sm_scale: Scale factor for attention scores
        b_start_loc: Starting indices for each batch
        b_seqlen: Sequence length for each batch
    """
    batch, heads, seqlen, d_model = q.shape
    
    # Output tensor
    out = torch.empty_like(q)
    
    # Block sizes
    BLOCK_M = 128
    BLOCK_DMODEL = d_model
    BLOCK_N = 128
    
    # Grid dimensions
    num_blocks = triton.cdiv(seqlen, BLOCK_M)
    grid = (batch * heads * num_blocks,)
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, out, b_start_loc, b_seqlen,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        window_size, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N,
    )
    
    return out
