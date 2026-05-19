import triton
import triton.language as tl

# Cache for storing compiled kernel
cached_kernel = None

@triton.jit
def _fwd_kernel(
    Q, K, V,                    # Query, Key, Value tensors
    sm_scale,                   # Scaling factor for attention scores
    B_Start_Loc, B_Seqlen,     # Batch sequence information
    Out,                        # Output tensor
    stride_qbs, stride_qh,      # Strides for Q tensor
    stride_kbs, stride_kh,      # Strides for K tensor
    stride_vbs, stride_vh,      # Strides for V tensor
    stride_obs, stride_oh,      # Strides for output tensor
    kv_group_num: tl.constexpr, # Number of key/value groups
    BLOCK_M: tl.constexpr,      # Block size for sequence dimension
    BLOCK_DMODEL: tl.constexpr, # Block size for model dimension
    BLOCK_N: tl.constexpr,      # Block size for attention dimension
):
    # Get current batch, head, and sequence block
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    
    # Calculate current KV head based on grouping
    cur_kv_head = cur_head // kv_group_num
    
    # Load batch sequence information
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    
    # Calculate starting position for this block
    block_start_loc = BLOCK_M * start_m
    
    # Initialize offset ranges for different dimensions
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # Calculate pointer offsets for Q, K, V tensors
    off_q = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + 
             cur_head * stride_qh + offs_d[None, :])
    off_k = (offs_n[None, :] * stride_kbs + 
             cur_kv_head * stride_kh + offs_d[:, None])
    off_v = (offs_n[:, None] * stride_vbs + 
             cur_kv_head * stride_vh + offs_d[None, :])
    
    # Load Q values with masking
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    
    # Initialize pointers for K and V
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    
    # Initialize accumulators for softmax computation
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Determine block mask based on sequence length
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    
    # Main attention computation loop
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K values with masking
        k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
                   mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
                   other=0.0)
        
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
        
        # Scale accumulator
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        
        # Load V values and compute output
        v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
                   mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
                   other=0.0)
        
        # Update output accumulator
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        
        # Update softmax accumulators
        l_i = l_i_new
        m_i = m_i_new
    
    # Store final output
    off_o = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + 
             cur_head * stride_oh + offs_d[None, :])
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)

def context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len):
    """
    Wrapper function for the context attention forward pass.
    
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
    
    # Verify dimensions
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}
    
    # Calculate scaling factor and batch/head dimensions
    sm_scale = 1.0 / (Lq ** 0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]
    
    # Configure grid and warps
    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if Lk <= 64 else 8
    
    # Use cached kernel if available
    global cached_kernel
    if cached_kernel:
        cached_kernel(grid, num_warps, q, k, v, sm_scale, b_start_loc, b_seq_len, o,
                     q.stride(0), q.stride(1), k.stride(0), k.stride(1),
                     v.stride(0), v.stride(1), o.stride(0), o.stride(1))
        return
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, sm_scale, b_start_loc, b_seq_len, o,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        v.stride(0), v.stride(1), o.stride(0), o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1
    )
    
    # Cache the kernel for future use
    cached_kernel = triton.runtime.JITFunction(_fwd_kernel.fn)
