import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    H: tl.constexpr, W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Program ID
    cur_batch = tl.program_id(0)  # Batch dimension
    cur_head = tl.program_id(1)   # Head dimension
    start_m = tl.program_id(2)    # Sequence length dimension
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Compute scale factor for attention scores
    scale = 1.0 / tl.sqrt(float(BLOCK_DMODEL))
    
    # Initialize pointers for Q, K, V
    q_ptrs = Q + (cur_batch * stride_qbs + cur_head * stride_qh + 
                  offs_m[:, None] * stride_qd + offs_d[None, :])
    k_ptrs = K + (cur_batch * stride_kbs + cur_head * stride_kh +
                  offs_n[None, :] * stride_kd + offs_d[:, None])
    v_ptrs = V + (cur_batch * stride_vbs + cur_head * stride_vh +
                  offs_n[:, None] * stride_vd + offs_d[None, :])
    
    # Load Q block
    q = tl.load(q_ptrs)
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Relative position indices
    rel_h = (offs_m[:, None] // W - offs_n[None, :] // W)
    rel_w = (offs_m[:, None] % W - offs_n[None, :] % W)
    
    # Main loop
    for start_n in range(0, H * W, BLOCK_N):
        # Load K, V blocks
        k = tl.load(k_ptrs + start_n * stride_kd)
        v = tl.load(v_ptrs + start_n * stride_vd)
        
        # Compute attention scores
        qk = tl.dot(q, k)
        qk = qk * scale
        
        # Add relative positional bias
        rel_pos_bias = tl.load(B0 + rel_h * H + rel_w)
        qk = qk + rel_pos_bias
        
        # Compute attention weights
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Update running statistics
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        
        # Update accumulator
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        acc += tl.dot(p, v)
        
        # Update statistics
        l_i = l_i_new
        m_i = m_i_new
    
    # Store output
    out_ptrs = Out + (cur_batch * stride_obs + cur_head * stride_oh +
                      offs_m[:, None] * stride_od + offs_d[None, :])
    tl.store(out_ptrs, acc)

def _attention_rel_h_rel_w_kernel_aligned_device(
    q, k, v, b0, out,
    h: int, w: int,
    block_size: int = 64
):
    batch_size = q.shape[0]
    num_heads = q.shape[1]
    seq_len = h * w
    d_model = q.shape[-1]
    
    # Configure kernel parameters
    BLOCK = block_size
    num_warps = 4 if d_model <= 64 else 8
    
    # Launch kernel
    grid = (batch_size, num_heads, triton.cdiv(seq_len, BLOCK))
    
    _fwd_kernel_aligned[grid](
        q, k, v, b0, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        H=h, W=w,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_DMODEL=d_model,
        num_warps=num_warps,
        num_stages=1
    )
