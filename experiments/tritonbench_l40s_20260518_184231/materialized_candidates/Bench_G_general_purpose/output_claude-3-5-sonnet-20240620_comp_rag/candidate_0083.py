import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_batch_head = tl.program_id(1)
    
    # Batch and head indexing
    batch_id = pid_batch_head // stride_qh
    head_id = pid_batch_head % stride_qh

    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Initialize pointers
    q_ptrs = Q + batch_id * stride_qbs + head_id * stride_qh + offs_m[:, None] * stride_qd + offs_d[None, :]
    k_ptrs = K + batch_id * stride_kbs + head_id * stride_kh + offs_n[None, :] * stride_kd + offs_d[:, None]
    v_ptrs = V + batch_id * stride_vbs + head_id * stride_vh + offs_n[:, None] * stride_vd + offs_d[None, :]
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Load Q block
    q = tl.load(q_ptrs)
    
    # Loop over K,V blocks
    for block_n in range(0, stride_kbs, BLOCK_N):
        # Load K,V blocks
        k = tl.load(k_ptrs + block_n * stride_kd)
        v = tl.load(v_ptrs + block_n * stride_vd)
        
        # Compute attention scores
        qk = tl.dot(q, k)
        qk = qk * sm_scale

        # Add bias if provided
        if B0 is not None:
            bias = tl.load(B0 + batch_id * stride_obs + head_id * stride_oh + 
                         offs_m[:, None] * stride_od + (block_n + offs_n)[None, :])
            qk = qk + bias

        # Compute softmax
        m_ij = tl.max(qk, 1)
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Update running max/sum
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - m_i_new)
        beta = tl.exp2(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        # Update accumulator
        p = p * (beta / l_i_new)[:, None]
        acc = acc * ((l_i / l_i_new) * alpha)[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        # Update m_i and l_i
        m_i = m_i_new
        l_i = l_i_new

    # Write output
    out_ptrs = Out + batch_id * stride_obs + head_id * stride_oh + offs_m[:, None] * stride_od + offs_d[None, :]
    tl.store(out_ptrs, acc)

def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, out):
    # Get input dimensions
    batch_size, num_heads, seq_len, d_model = q.shape
    
    # Validate input shapes
    assert k.shape == (batch_size, num_heads, seq_len, d_model)
    assert v.shape == (batch_size, num_heads, seq_len, d_model)
    if b0 is not None:
        assert b0.shape == (batch_size, num_heads, seq_len, seq_len)
    
    # Set block sizes based on d_model
    BLOCK_M = 128
    BLOCK_N = 128 
    BLOCK_DMODEL = d_model

    # Calculate grid dimensions
    grid = (
        triton.cdiv(seq_len, BLOCK_M), 
        batch_size * num_heads
    )

    # Calculate scaling factor
    sm_scale = 1.0 / (d_model ** 0.5)

    # Launch kernel
    _fwd_kernel_aligned[grid](
        q, k, v, b0, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=8,
        num_stages=2
    )
    
    return out
