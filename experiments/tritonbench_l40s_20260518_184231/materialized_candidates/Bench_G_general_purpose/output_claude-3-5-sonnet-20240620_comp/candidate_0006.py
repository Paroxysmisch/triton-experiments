import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_b0z, stride_b0h, stride_b0m, stride_b0n,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX, P_SEQ, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    sm_scale,
    OUT_DTYPE: tl.constexpr
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_z = tl.program_id(2)
    pid_h = tl.program_id(3)

    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # Initialize pointers
    q_ptrs = Q + (pid_z * stride_qz + pid_h * stride_qh + 
                  offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_ptrs = K + (pid_z * stride_kz + pid_h * stride_kh + 
                  offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk)
    v_ptrs = V + (pid_z * stride_vz + pid_h * stride_vh + 
                  offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    
    # Load Q block
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=OUT_DTYPE)
    
    # Load B0 bias if provided
    if B0 is not None:
        b0_ptrs = B0 + (pid_z * stride_b0z + pid_h * stride_b0h +
                        offs_m[:, None] * stride_b0m + offs_n[None, :] * stride_b0n)
        bias = tl.load(b0_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX + P_SEQ), other=0.0)
    
    # Initialize softmax tracking
    m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Iterate over sequence length
    for start_n in range(0, N_CTX + P_SEQ, BLOCK_N):
        # Load K block
        k = tl.load(k_ptrs, mask=(start_n + offs_n[None, :]) < (N_CTX + P_SEQ), other=0.0)
        
        # Compute attention scores
        scores = tl.dot(q, k) * sm_scale
        if B0 is not None:
            scores = scores + bias
            
        # Update softmax stats
        m_next = tl.maximum(m, tl.max(scores, 1))
        l_next = l * tl.exp(m[:, None] - m_next[:, None]) + tl.sum(tl.exp(scores - m_next[:, None]), 1)
        
        # Load V block and compute weighted sum
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < (N_CTX + P_SEQ), other=0.0)
        acc += tl.dot(tl.exp(scores - m_next[:, None]), v)
        
        # Update softmax stats
        m = m_next
        l = l_next
        
        # Update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        if B0 is not None:
            b0_ptrs += BLOCK_N * stride_b0n
            
    # Normalize accumulator
    acc = acc / l[:, None]
    
    # Write output
    out_ptrs = Out + (pid_z * stride_oz + pid_h * stride_oh +
                      offs_m[:, None] * stride_om + offs_k[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)

def attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, sm_scale):
    batch_size, n_heads, seq_len, d_head = q.shape
    p_seq = k.shape[2] - seq_len  # Additional sequence length for relative positions
    
    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = d_head
    
    # Compute grid dimensions
    grid = (
        triton.cdiv(seq_len, BLOCK_M),  # Sequence blocks
        triton.cdiv(seq_len + p_seq, BLOCK_N),  # Extended sequence blocks
        batch_size,  # Batch dimension
        n_heads,  # Head dimension
    )
    
    # Output tensor
    output = torch.empty_like(q)
    
    # Launch kernel
    _fwd_kernel_aligned[grid](
        q, k, v, b0, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b0.stride(0) if b0 is not None else 0,
        b0.stride(1) if b0 is not None else 0,
        b0.stride(2) if b0 is not None else 0,
        b0.stride(3) if b0 is not None else 0,
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch_size, n_heads, seq_len, p_seq,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        sm_scale=sm_scale,
        OUT_DTYPE=output.dtype,
        num_warps=4,
        num_stages=2
    )
    
    return output
