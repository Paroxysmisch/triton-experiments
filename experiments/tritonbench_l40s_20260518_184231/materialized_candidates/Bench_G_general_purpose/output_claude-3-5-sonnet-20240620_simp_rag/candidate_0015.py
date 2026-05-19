import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    B_Start_Loc, B_Seqlen,
    sm_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    batch_size, num_heads, seqlen, 
    BLOCK: tl.constexpr
):
    # Program ID
    bid = tl.program_id(0)
    
    # Calculate batch and head index
    batch_id = bid // num_heads
    head_id = bid % num_heads

    # Get sequence info for this batch
    start_idx = tl.load(B_Start_Loc + batch_id)
    seq_len = tl.load(B_Seqlen + batch_id)

    # Initialize pointers
    q_offset = batch_id * stride_qb + head_id * stride_qh
    k_offset = batch_id * stride_kb + head_id * stride_kh
    v_offset = batch_id * stride_vb + head_id * stride_vh
    o_offset = batch_id * stride_ob + head_id * stride_oh

    # Initialize accumulators
    acc_o = tl.zeros([BLOCK], dtype=tl.float32)
    acc_s = tl.zeros([BLOCK], dtype=tl.float32)

    # Block-level loop
    for block_start in range(0, seq_len, BLOCK):
        block_end = min(block_start + BLOCK, seq_len)
        block_size = block_end - block_start

        # Load query block
        q = tl.load(Q + q_offset + (start_idx + block_start) * stride_qm + tl.arange(0, BLOCK))
        
        # Initialize max value for numerical stability
        m = -float('inf')

        # Key-Value loop
        for k_block_start in range(0, seq_len, BLOCK):
            k_block_end = min(k_block_start + BLOCK, seq_len)
            
            # Load key and value blocks
            k = tl.load(K + k_offset + (start_idx + k_block_start) * stride_kn + tl.arange(0, BLOCK))
            v = tl.load(V + v_offset + (start_idx + k_block_start) * stride_vn + tl.arange(0, BLOCK))

            # Compute attention scores
            qk = tl.dot(q, k) * sm_scale
            
            # Apply causal mask
            mask = tl.arange(0, BLOCK)[:, None] >= tl.arange(0, BLOCK)[None, :]
            qk = tl.where(mask, qk, float('-inf'))
            
            # Update max value
            m_prev = m
            m = tl.maximum(m, tl.max(qk, 1))
            
            # Compute attention weights
            p = tl.exp(qk - m[:, None])
            
            # Update accumulators
            acc_o = acc_o * tl.exp(m_prev - m)[:, None] + tl.dot(p, v)
            acc_s = acc_s * tl.exp(m_prev - m) + tl.sum(p, 1)

        # Normalize and store output
        out = acc_o / acc_s[:, None]
        tl.store(Out + o_offset + (start_idx + block_start) * stride_om + tl.arange(0, BLOCK), out)

def context_attention_fwd(q, k, v, sm_scale, b_start_loc, b_seqlen):
    batch_size, num_heads, max_seqlen, head_dim = q.shape
    
    # Output tensor
    out = torch.empty_like(q)
    
    # Launch configs
    BLOCK = 128
    grid = (batch_size * num_heads,)
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, out,
        b_start_loc, b_seqlen,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        batch_size, num_heads, max_seqlen,
        BLOCK=BLOCK
    )
    
    return out
