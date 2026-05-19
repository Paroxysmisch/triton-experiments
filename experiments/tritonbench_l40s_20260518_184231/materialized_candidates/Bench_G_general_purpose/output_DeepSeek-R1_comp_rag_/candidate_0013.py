import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    B_Start_Loc, B_Seqlen, Out,
    stride_qbs, stride_qh,
    stride_kbs, stride_kh,
    stride_vbs, stride_vh,
    stride_obs, stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 3D grid: batch, head, sequence-block
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num  # Grouped query attention
    seq_len = tl.load(B_Seqlen + cur_batch)
    start_idx = tl.load(B_Start_Loc + cur_batch)

    # Offsets for block processing
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Q block with masking for valid sequence length
    q_ptrs = Q + (start_idx + offs_m[:, None])*stride_qbs + cur_head*stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)

    # Initialize accumulation variables for online softmax
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Process key/value blocks in causal order
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # Load K block
        k_ptrs = K + (start_idx + start_n)*stride_kbs + cur_kv_head*stride_kh + offs_d[:, None]
        k = tl.load(k_ptrs, mask=(start_n + offs_n[None, :]) < seq_len, other=0.0)
        
        # Compute QK^T with causal masking
        qk = tl.dot(q, k) * sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float('-inf'))
        
        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new[:, None])
        l_new = alpha * l_i + tl.sum(beta, axis=1)
        
        # Update attention accumulation
        acc_scale = l_i / l_new * alpha
        acc *= acc_scale[:, None]
        p = beta / l_new[:, None]
        
        # Load V and accumulate
        v_ptrs = V + (start_idx + start_n)*stride_vbs + cur_kv_head*stride_vh + offs_d[None, :]
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seq_len, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        
        m_i, l_i = m_new, l_new

    # Store final output
    out_ptrs = Out + (start_idx + offs_m[:, None])*stride_obs + cur_head*stride_oh + offs_d[None, :]
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < seq_len)

def context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_len):
    BLOCK = 128 if torch.cuda.get_device_capability()[0] >= 8 else 64
    d_model = q.size(-1)
    assert d_model in {16, 32, 64, 128, 256}
    
    # Configure kernel launch parameters
    sm_scale = 1.0 / (d_model ** 0.5)
    batch, head = b_seq_len.size(0), q.size(1)
    kv_group_num = q.size(1) // k.size(1)
    
    grid = (batch, head, triton.cdiv(max_len, BLOCK))
    num_warps = 4 if d_model <= 64 else 8
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, sm_scale,
        b_start_loc, b_seq_len, o,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=d_model,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1
    )
