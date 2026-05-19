import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    q_ptr, k_ptr, cos_ptr, sin_ptr, 
    output_q_ptr, output_k_ptr,
    stride_qm, stride_ql, stride_km, stride_kl,
    stride_om, stride_ol, stride_okm, stride_okl,
    N, M, D, BACKWARD_PASS: tl.constexpr
):
    # Define block size
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_D = 64
    
    # Calculate block indices
    m_block = tl.program_id(0)
    d_block = tl.program_id(1)
    
    # Calculate offsets
    m_offset = m_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    d_offset = d_block * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    
    # Mask to handle out-of-bounds accesses
    mask_m = m_offset < M
    mask_d = d_offset < D
    
    # Load data from global memory
    q = tl.load(q_ptr + m_offset[:, None] * stride_qm + d_offset[None, :] * stride_ql, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    k = tl.load(k_ptr + m_offset[:, None] * stride_km + d_offset[None, :] * stride_kl, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    
    # Load cosine and sine values
    cos = tl.load(cos_ptr + d_offset, mask=mask_d, other=1.0)
    sin = tl.load(sin_ptr + d_offset, mask=mask_d, other=0.0)
    
    # Apply rotation-based positional encoding
    if BACKWARD_PASS:
        q_rot = q * cos[None, :] - k * sin[None, :]
        k_rot = k * cos[None, :] + q * sin[None, :]
    else:
        q_rot = q * cos[None, :] + k * sin[None, :]
        k_rot = k * cos[None, :] - q * sin[None, :]
    
    # Store results back to global memory
    tl.store(output_q_ptr + m_offset[:, None] * stride_om + d_offset[None, :] * stride_ol, q_rot, mask=mask_m[:, None] & mask_d[None, :])
    tl.store(output_k_ptr + m_offset[:, None] * stride_okm + d_offset[None, :] * stride_okl, k_rot, mask=mask_m[:, None] & mask_d[None, :])

def rope_backward(q, k, cos, sin, stride_q, stride_k, stride_cos, stride_sin):
    # Get dimensions
    M, D = q.shape
    N = k.shape[0]
    
    # Prepare output buffers
    output_q = torch.empty_like(q)
    output_k = torch.empty_like(k)
    
    # Launch Triton kernel
    grid = (triton.cdiv(M, 128), triton.cdiv(D, 64))
    _triton_rope[grid](
        q, k, cos, sin,
        output_q, output_k,
        stride_q[0], stride_q[1],
        stride_k[0], stride_k[1],
        output_q.stride(0), output_q.stride(1),
        output_k.stride(0), output_k.stride(1),
        N, M, D, BACKWARD_PASS=True
    )
    
    return output_q, output_k
