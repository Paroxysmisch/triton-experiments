import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out,
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_b_b, stride_b_h, stride_b_m, stride_b_n,
    stride_out_b, stride_out_h, stride_out_m, stride_out_d,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    # Offsets for Q block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)

    # Pointers for Q block
    q_ptrs = Q + pid_batch * stride_q_b + pid_head * stride_q_h + \
            (offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d)
    q_mask = offs_m[:, None] < tl.program_id(2) * BLOCK_M + BLOCK_M  # Simplified mask
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    q = q * sm_scale

    # Initialize accumulators
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Loop over K/V blocks
    num_k_blocks = tl.cdiv(tl.num_programs(2), BLOCK_N)  # Simplified loop count
    for k_idx in range(num_k_blocks):
        # Compute K block offsets
        offs_k = k_idx * BLOCK_N + offs_n
        
        # Load K block
        k_ptrs = K + pid_batch * stride_k_b + pid_head * stride_k_h + \
                (offs_k[None, :] * stride_k_n + offs_d[:, None] * stride_k_d)
        k = tl.load(k_ptrs, mask=offs_k[None, :] < num_k_blocks * BLOCK_N, other=0.0)
        
        # Compute QK^T
        s = tl.dot(q, k, allow_tf32=True)
        
        # Add bias if present
        if HAS_BIAS:
            b_ptrs = B0 + pid_batch * stride_b_b + pid_head * stride_b_h + \
                    (offs_m[:, None] * stride_b_m + offs_k[None, :] * stride_b_n)
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < BLOCK_M) & (offs_k[None, :] < num_k_blocks * BLOCK_N), other=0.0)
            s += b

        # Online softmax update
        m_curr = tl.maximum(tl.max(s, 1), m_prev)
        alpha = tl.exp2(m_prev - m_curr)
        l_curr = tl.exp2(m_prev - m_curr) * l_prev + tl.sum(tl.exp2(s - m_curr[:, None]), 1)
        
        # Load V block and update acc
        v_ptrs = V + pid_batch * stride_v_b + pid_head * stride_v_h + \
                (offs_k[:, None] * stride_v_n + offs_d[None, :] * stride_v_d)
        v = tl.load(v_ptrs, mask=offs_k[:, None] < num_k_blocks * BLOCK_N, other=0.0)
        
        acc = acc * alpha[:, None] + tl.dot(tl.exp2(s - m_curr[:, None]), v, allow_tf32=True)
        
        # Update previous values
        m_prev = m_curr
        l_prev = l_curr

    # Normalize and write output
    acc = acc / l_prev[:, None]
    out_ptrs = Out + pid_batch * stride_out_b + pid_head * stride_out_h + \
              (offs_m[:, None] * stride_out_m + offs_d[None, :] * stride_out_d)
    tl.store(out_ptrs, acc, mask=q_mask)

def _attention_rel_h_rel_w_kernel_aligned_device(
    q, k, v, rel_h_w, output, sm_scale,
    BLOCK_M=64, BLOCK_N=64, BLOCK_DMODEL=64
):
    # Validate inputs
    assert q.dtype == k.dtype == v.dtype, "Input types must match"
    assert rel_h_w is None or rel_h_w.dtype == q.dtype, "Bias type mismatch"
    
    batch, heads, seq_len, d_model = q.shape
    grid = (batch, heads, triton.cdiv(seq_len, BLOCK_M))
    
    # Kernel configuration
    config = {
        'BLOCK_M': BLOCK_M,
        'BLOCK_N': BLOCK_N,
        'BLOCK_DMODEL': BLOCK_DMODEL,
        'HAS_BIAS': rel_h_w is not None,
        'num_warps': 4,
        'num_stages': 3
    }
    
    # Prepare bias parameters
    b0 = rel_h_w if rel_h_w is not None else q.new_empty(0)
    strides = (
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b0.stride(0), b0.stride(1), b0.stride(2), b0.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3)
    )
    
    # Launch kernel
    _fwd_kernel_aligned[grid](
        q, k, v, b0, output,
        *strides,
        sm_scale=sm_scale,
        **config
    )
