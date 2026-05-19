import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att1(
    Q, K,
    B_Loc, B_Start_Loc, B_Seqlen,
    Att_Out,
    stride_qbs, stride_qh,
    stride_kbs, stride_kh,
    stride_obs, stride_oh,
    max_input_len,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Program ID for the current instance
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    # Load batch sequence information
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start = tl.load(B_Start_Loc + cur_batch)
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Calculate memory offsets for Q
    q_offset = (
        (cur_batch_start + offs_m[:, None]) * stride_qbs +
        cur_head * stride_qh +
        offs_d[None, :]
    )
    
    # Load Q block
    q = tl.load(
        Q + q_offset,
        mask=offs_m[:, None] < cur_batch_seq_len,
        other=0.0
    )
    
    # Initialize accumulators for softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Process key blocks
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        # Calculate K offsets and load K block
        k_offset = (
            (cur_batch_start + start_n + offs_n[None, :]) * stride_kbs +
            cur_head * stride_kh +
            offs_d[:, None]
        )
        
        k = tl.load(
            K + k_offset,
            mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
            other=0.0
        )
        
        # Compute attention scores
        qk = tl.dot(q, k) * sm_scale
        
        # Apply causal mask
        qk = tl.where(
            offs_m[:, None] >= (start_n + offs_n[None, :]),
            qk,
            float("-inf")
        )
        
        # Compute local softmax values
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Update accumulators
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        
        # Update attention output
        p_scale = beta / l_i_new
        acc = acc * (l_i / l_i_new * alpha)[:, None] + p * p_scale[:, None]
        
        # Update running max and sum
        l_i = l_i_new
        m_i = m_i_new
    
    # Store results
    out_offset = (
        (cur_batch_start + offs_m[:, None]) * stride_obs +
        cur_head * stride_oh +
        offs_n[None, :]
    )
    
    tl.store(
        Att_Out + out_offset,
        acc,
        mask=offs_m[:, None] < cur_batch_seq_len
    )

def token_att_fwd(q, k, b_loc, b_start_loc, b_seq_len, max_input_len):
    # Determine block size based on GPU capability
    BLOCK = 128 if triton.runtime.driver.utils.get_device_capability()[0] >= 8 else 64
    
    # Get dimensions
    batch = b_seq_len.shape[0]
    head = q.shape[1]
    dim = q.shape[-1]
    
    # Calculate scaling factor
    sm_scale = 1.0 / (dim ** 0.5)
    
    # Initialize output tensor
    att_out = torch.empty_like(q)
    
    # Configure grid and launch kernel
    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if dim <= 64 else 8
    
    _fwd_kernel_token_att1[grid](
        q, k,
        b_loc, b_start_loc, b_seq_len,
        att_out,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        att_out.stride(0), att_out.stride(1),
        max_input_len,
        sm_scale,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_DMODEL=dim,
        num_warps=num_warps,
        num_stages=1
    )
    
    return att_out
