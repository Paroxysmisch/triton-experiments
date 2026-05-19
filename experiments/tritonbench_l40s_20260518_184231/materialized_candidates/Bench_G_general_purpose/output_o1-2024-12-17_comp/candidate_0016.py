import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    B_Start_Loc_ptr, B_Seqlen_ptr,
    Out_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    B, H, M, D,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr
):
    # -------------------------------------------------------------------------
    #  Compute indices for (batch, head) and partial M-block in the sequence
    # -------------------------------------------------------------------------
    bh = tl.program_id(0)  # combined (batch, head) dimension
    head_idx = bh % H
    batch_idx = bh // H
    block_m_idx = tl.program_id(1)
    
    # Offsets for Q in the M dimension
    m_start = block_m_idx * BLOCK_M
    m_off = m_start + tl.arange(0, BLOCK_M)
    
    # Pointers for Q
    q_ptrs = Q_ptr + batch_idx * stride_qb \
                      + head_idx * stride_qh \
                      + (m_off[:, None] * stride_qm) \
                      + (tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qd)
    
    # Load Q block [BLOCK_M, BLOCK_DMODEL]
    q = tl.load(q_ptrs, mask=(m_off[:, None] < M), other=0.0)
    
    # Sequence info
    seq_start_loc = tl.load(B_Start_Loc_ptr + batch_idx)
    seq_len = tl.load(B_Seqlen_ptr + batch_idx)
    
    # Sliding window boundaries (example definition; adapt as needed)
    # left_idx and right_idx define how many blocks of K/V we attend to.
    # This could be computed or passed in depending on your strategy.
    left_idx = tl.max_seq(0, m_start - BLOCK_N)      # Example for left boundary
    right_idx = tl.min(seq_len, m_start + BLOCK_M*2) # Example for right boundary
    
    # Initialize accumulators for partial attention outputs
    out_accum = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    max_scores = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    sum_scores = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # Loop over blocks of K and V within the sliding window
    n_block = left_idx
    while n_block < right_idx:
        # ---------------------------------------------------------------------
        # 1) Load K block
        # ---------------------------------------------------------------------
        n_off = n_block + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * stride_kb \
                        + head_idx * stride_kh \
                        + (n_off[None, :] * stride_kn) \
                        + (tl.arange(0, BLOCK_DMODEL)[:, None] * stride_kd)
        k = tl.load(k_ptrs, mask=(n_off[None, :] < seq_len), other=0.0)
        
        # ---------------------------------------------------------------------
        # 2) Compute QK^T for attention scores
        # ---------------------------------------------------------------------
        # q: [BLOCK_M, BLOCK_DMODEL], k: [BLOCK_DMODEL, BLOCK_N]
        # => qk: [BLOCK_M, BLOCK_N]
        qk = tl.matmul(q, k)
        
        # Scale scores
        qk *= sm_scale
        
        # ---------------------------------------------------------------------
        # 3) Numerically stable accumulation (max-sum)
        # ---------------------------------------------------------------------
        current_max = tl.maximum(tl.max(qk, 1), max_scores)
        exp_old = tl.exp(max_scores - current_max)
        exp_new = tl.exp(qk - current_max[:, None])
        
        # Update sum of scores
        sum_scores = sum_scores * exp_old + tl.sum(exp_new, 1)
        max_scores = current_max
        
        # ---------------------------------------------------------------------
        # 4) Load V block and accumulate partial output
        # ---------------------------------------------------------------------
        v_ptrs = V_ptr + batch_idx * stride_vb \
                        + head_idx * stride_vh \
                        + (n_off[None, :] * stride_vn) \
                        + (tl.arange(0, BLOCK_DMODEL)[:, None] * stride_vd)
        v = tl.load(v_ptrs, mask=(n_off[None, :] < seq_len), other=0.0)
        
        # Weighted accumulation
        # qk: [BLOCK_M, BLOCK_N], v: [BLOCK_DMODEL, BLOCK_N]
        # We do exponent on qk in a numerically stable manner
        w_new = exp_new / sum_scores[:, None]
        out_accum = out_accum * exp_old[:, None] + tl.matmul(w_new, tl.trans(v))
        
        n_block += BLOCK_N
    
    # -------------------------------------------------------------------------
    # Write output
    # -------------------------------------------------------------------------
    # out_accum is [BLOCK_M, BLOCK_DMODEL]
    out_ptrs = Out_ptr + batch_idx * stride_ob \
                         + head_idx * stride_oh \
                         + (m_off[:, None] * stride_om) \
                         + (tl.arange(0, BLOCK_DMODEL)[None, :] * stride_od)
    
    # Store final scale
    tl.store(out_ptrs, out_accum, mask=(m_off[:, None] < M))

def context_attention_fwd(Q, K, V, B_Start_Loc, B_Seqlen, sm_scale, Out):
    """
    Q, K, V: [B, H, M, D]
    B_Start_Loc: [B]
    B_Seqlen: [B]
    Out: [B, H, M, D]
    sm_scale: scalar
    """
    B, H, M, D = Q.shape
    # Define block sizes
    BLOCK_M = 64
    BLOCK_DMODEL = 64
    BLOCK_N = 64
    
    grid = (B * H, (M + BLOCK_M - 1) // BLOCK_M)
    
    _fwd_kernel[grid](
        Q, K, V,
        B_Start_Loc, B_Seqlen,
        Out,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        B, H, M, D,
        BLOCK_M, BLOCK_DMODEL, BLOCK_N
    )
    return Out
