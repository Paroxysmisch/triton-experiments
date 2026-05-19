import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    B_Start_Loc, B_Seqlen,
    Out,
    stride_qbs, stride_qh,
    stride_kbs, stride_kh,
    stride_vbs, stride_vh,
    stride_obs, stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Thread block indices
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    # Adjust for grouped query attention
    cur_kv_head = cur_head // kv_group_num

    # Load sequence metadata
    cur_seq_len = tl.load(B_Seqlen + cur_batch)
    seq_start = tl.load(B_Start_Loc + cur_batch)
    
    # Calculate block boundaries
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Initialize query pointer
    q_offs = (seq_start + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :]
    q = tl.load(Q + q_offs, mask=offs_m[:, None] < cur_seq_len, other=0.0)

    # Initialize accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Process chunks of K/V
    loop_upper = tl.minimum((start_m + 1) * BLOCK_M, cur_seq_len)
    for start_n in range(0, loop_upper, BLOCK_N):
        # Load K block
        k_offs = (seq_start + start_n + offs_n[None, :]) * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
        k = tl.load(K + k_offs, mask=(start_n + offs_n[None, :]) < cur_seq_len, other=0.0)
        
        # Compute QK^T
        qk = tl.dot(q, k, allow_tf32=False)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float('-inf'))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        
        # Update statistics
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        l_new = alpha * l_i + beta * l_ij

        # Update accumulator
        p_scale = beta / l_new
        acc_scale = l_i / l_new * alpha
        acc *= acc_scale[:, None]
        
        # Load V and accumulate
        v_offs = (seq_start + start_n + offs_n[:, None]) * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]
        v = tl.load(V + v_offs, mask=(start_n + offs_n[:, None]) < cur_seq_len, other=0.0)
        acc += tl.dot((p * p_scale[:, None]).to(v.dtype), v)

        # Update state
        m_i = m_new
        l_i = l_new

    # Write output
    out_offs = (seq_start + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :]
    tl.store(Out + out_offs, acc, mask=offs_m[:, None] < cur_seq_len)


def context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_seq_len):
    # Determine kernel configuration
    d_model = q.size(-1)
    device = q.device
    
    BLOCK = 128 if torch.cuda.get_device_capability(device) >= (8, 0) else 64
    BLOCK_M = BLOCK_N = BLOCK
    num_warps = 8 if d_model >= 128 else 4
    
    # Validate inputs
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "Feature dimension mismatch"
    assert q.shape[1] % k.shape[1] == 0, "Query heads must be multiple of
