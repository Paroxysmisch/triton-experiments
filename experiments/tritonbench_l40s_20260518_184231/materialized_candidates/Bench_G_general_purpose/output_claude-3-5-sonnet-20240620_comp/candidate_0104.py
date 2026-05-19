import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i,
    q, k, v,
    q_scale, k_scale,
    start_n, num_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    # Pointers for the current block
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load Q block
    q_ptrs = q + offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :]
    q_block = tl.load(q_ptrs)
    q_block = q_block * q_scale
    
    # Initialize accumulators
    block_m_i = m_i
    block_l_i = l_i
    block_acc = acc
    
    # Process K,V blocks
    for n in range(start_n, start_n + num_n, BLOCK_N):
        k_ptrs = k + (n + offs_n[:, None]) * BLOCK_DMODEL + offs_d[None, :]
        v_ptrs = v + (n + offs_n[:, None]) * BLOCK_DMODEL + offs_d[None, :]
        
        # Load K,V blocks
        k_block = tl.load(k_ptrs) * k_scale
        v_block = tl.load(v_ptrs)
        
        # Compute attention scores
        scores = tl.dot(q_block, tl.trans(k_block))
        
        # Update running maximum
        m_ij = tl.max(scores, 1)
        m_i_new = tl.maximum(block_m_i, m_ij)
        alpha = tl.exp(block_m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        
        # Update accumulator
        block_acc = block_acc * alpha[:, None]
        block_l_i = block_l_i * alpha + tl.sum(beta, 1)
        
        # Compute attention weights
        p = beta / block_l_i[:, None]
        
        # Update weighted sum
        block_acc += tl.dot(p, v_block)
        block_m_i = m_i_new
        
    return block_acc, block_l_i, block_m_i

@triton.jit
def _attn_fwd(
    q, k, v, o,
    q_scale, k_scale,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    num_heads = stride_qh // stride_qm
    head_idx = pid // (stride_qm // BLOCK_M)
    block_idx = pid % (stride_qm // BLOCK_M)
    
    # Initialize pointers
    offs_m = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    
    # Compute attention for this block
    q_ptr = q + head_idx * stride_qh + offs_m[:, None] * stride_qd + offs_d[None, :]
    k_ptr = k + head_idx * stride_kh
    v_ptr = v + head_idx * stride_vh
    
    # Process blocks
    num_blocks = stride_kn // BLOCK_N
    for block_start_n in range(0, stride_kn, BLOCK_N):
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i,
            q_ptr, k_ptr, v_ptr,
            q_scale, k_scale,
            block_start_n, min(BLOCK_N, stride_kn - block_start_n),
            BLOCK_M, BLOCK_N, BLOCK_DMODEL
        )
    
    # Write output
    acc = acc / l_i[:, None]
    o_ptr = o + head_idx * stride_oh + offs_m[:, None] * stride_od + offs_d[None, :]
    tl.store(o_ptr, acc)

def attention_forward(q, k, v, q_scale, k_scale):
    batch_size, num_heads, seq_len, d_model = q.shape
    
    # Allocate output
    o = torch.empty_like(q)
    
    # Launch kernel
    grid = (num_heads * triton.cdiv(seq_len, 128),)
    _attn_fwd[grid](
        q, k, v, o,
        q_scale, k_scale,
        seq_len, seq_len * num_heads, d_model,
        seq_len, seq_len * num_heads, d_model,
        seq_len, seq_len * num_heads, d_model,
        seq_len, seq_len * num_heads, d_model,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_DMODEL=d_model,
    )
    
    return o
