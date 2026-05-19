@triton.jit
def fused_chunk_retention_fwd_kernel(
    q, k, v, o, initial_state, final_state,
    s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    # Get program IDs for parallel execution
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    
    # Initialize block-level variables
    o_i = tl.arange(0, BT)
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    
    # Calculate decay factors
    d_b = tl.math.exp2(BT * b_b)
    d_o = tl.math.exp2((o_i + 1) * b_b)
    d_h = tl.math.exp2((BT - o_i - 1) * b_b)
    
    # Create mask and scaling matrix
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
    
    # Main computation loop
    for i in range(0, tl.cdiv(T, BT)):
        # Load blocks
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        
        # Scale query
        b_q = (b_q * scale).to(b_k.dtype)
        
        # Compute attention scores and output
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_s
        b_o = tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        
        # Update hidden states
        if CHECK and i == 0:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False) * d_o[:, None]
            b_h = d_b * b_h + tl.dot(b_k, (b_v * d_h[:, None]).to(b_k.dtype), allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False) * d_o[:, None]
            b_h = d_b * b_h + tl.dot(b_k, (b_v * d_h[:, None]).to(b_k.dtype), allow_tf32=False)

@triton.jit
def fused_chunk_retention_bwd_kernel(
    q, k, v, do, dq, dk, dv, initial_state,
    s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d,
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    # Similar initialization as forward kernel
    
    # Compute gradients for query, key and value
    for i in range(0, tl.cdiv(T, BT)):
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = (b_ds * d_s).to(b_k.dtype)
        b_dq = tl.dot(b_ds, b_k, allow_tf32=False)
        
        # Update gradients with hidden state contribution
        if CHECK and i == 0:
            b_dq += tl.dot(b_dd, b_h.to(b_k.dtype), allow_tf32=False)
            b_h = d_b * b_h + tl.dot((b_v * d_k[None, :]).to(b_k.dtype), b_k, allow_tf32=False)
        else:
            b_dq += tl.dot(b_dd, b_h.to(b_k.dtype), allow_tf32=False)
            b_h = d_b * b_h + tl.dot((b_v * d_k[None, :]).to(b_k.dtype), b_k, allow_tf32=False)
