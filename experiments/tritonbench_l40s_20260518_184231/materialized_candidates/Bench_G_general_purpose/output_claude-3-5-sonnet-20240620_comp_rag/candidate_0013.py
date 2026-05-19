import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V,                    # Query, Key, Value tensors
    sm_scale,                   # Scaling factor for attention scores
    B_Start_Loc,               # Starting locations for each batch
    B_Seqlen,                  # Sequence lengths for each batch
    Out,                       # Output tensor
    stride_qbs, stride_qh,     # Strides for Q tensor
    stride_kbs, stride_kh,     # Strides for K tensor
    stride_vbs, stride_vh,     # Strides for V tensor
    stride_obs, stride_oh,     # Strides for output tensor
    kv_group_num: tl.constexpr,# Number of query heads per kv head
    BLOCK_M: tl.constexpr,     # Block size for sequence dimension
    BLOCK_DMODEL: tl.constexpr,# Hidden dimension size
    BLOCK_N: tl.constexpr,     # Block size for key/value dimension
):
    # Get program ID for batch, head and sequence block
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    
    # Calculate current KV head considering head grouping
    cur_kv_head = cur_head // kv_group_num
    
    # Load batch sequence length and start location
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start = tl.load(B_Start_Loc + cur_batch)
    
    # Initialize offsets for parallel processing
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Calculate memory offsets for Q, K, V tensors
    q_offset = (
        (cur_batch_start + offs_m[:, None]) * stride_qbs +
        cur_head * stride_qh +
        offs_d[None, :]
    )
    k_offset = (
        offs_n[None, :] * stride_kbs +
        cur_kv_head * stride_kh +
        offs_d[:, None]
    )
    v_offset = (
        offs_n[:, None] * stride_vbs +
        cur_kv_head * stride_vh +
        offs_d[None, :]
    )
    
    # Load query block
    q = tl.load(Q + q_offset, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    
    # Initialize accumulators for softmax and output
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Calculate block mask for causal attention
    block_mask = tl.where(start_m * BLOCK_M < cur_batch_seq_len, 1, 0)
    
    # Main loop over key/value blocks
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load key block
        k = tl.load(
            K + k_offset + (cur_batch_start + start_n) * stride_kbs,
            mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
            other=0.0
        )
        
        # Compute attention scores
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        
        # Apply causal mask
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        
        # Compute softmax values
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Update softmax accumulators
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        
        # Scale attention weights
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        
        # Scale and update output accumulator
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        
        # Load value block and compute weighted sum
        v = tl.load(
            V + v_offset + (cur_batch_start + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
            other=0.0
        )
        
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        
        # Update running statistics
        l_i = l_i_new
        m_i = m_i_new
    
    # Write output
    out_offset = (
        (cur_batch_start + offs_m[:, None]) * stride_obs +
        cur_head * stride_oh +
        offs_d[None, :]
    )
    tl.store(Out + out_offset, acc, mask=offs_m[:, None] < cur_batch_seq_len)

def context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len):
    """
    Wrapper function to launch the context attention kernel
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        o: Output tensor
        b_start_loc: Batch start locations
        b_seq_len: Batch sequence lengths
        max_input_len: Maximum input sequence length
    """
    # Set block size based on GPU capability
    BLOCK = 128 if triton.runtime.driver.utils.CUDA_CAPABILITY[0] >= 8 else 64
    
    # Validate input dimensions
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}
    
    # Calculate attention scaling factor
    sm_scale = 1.0 / (Lq ** 0.5)
    
    # Get batch and head dimensions
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]
    
    # Configure kernel launch parameters
    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1
    )
