import triton
import triton.language as tl
import torch

@triton.jit
def rms_matmul_rbe(
    x_ptr,
    w_ptr,
    rms_w_ptr,
    output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_outm, stride_outn,
    THETA: tl.constexpr,
    apply_rotary: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    eps: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = m_offset < M
    mask_n = n_offset < N
    
    sum_squares = tl.zeros(BLOCK_SIZE_M, dtype=tl.float32)
    
    # Compute sum of squares for RMS
    for k in range(0, K, BLOCK_SIZE_K):
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offset < K
        
        x_ptrs = x_ptr + m_offset[:, None] * stride_xm + k_offset[None, :] * stride_xk
        x_vals = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        sum_squares += tl.sum(x_vals * x_vals, axis=1)
    
    rms = tl.sqrt((sum_squares / K) + eps)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Matrix multiplication with normalized values
    for k in range(0, K, BLOCK_SIZE_K):
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offset < K
        
        x_ptrs = x_ptr + m_offset[:, None] * stride_xm + k_offset[None, :] * stride_xk
        x_vals = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        normalized_x = x_vals / rms[:, None]
        
        rms_w_vals = tl.load(rms_w_ptr + k_offset, mask=mask_k, other=1.0)
        scaled_x = normalized_x * rms_w_vals[None, :]
        
        w_ptrs = w_ptr + k_offset[:, None] * stride_wk + n_offset[None, :] * stride_wn
        w_vals = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        acc += tl.dot(scaled_x, w_vals)
    
    # Apply rotary embeddings if enabled
    if apply_rotary:
        position = m_offset
        freq = 1.0 / (THETA ** (2 * (n_offset % (N//2)) / N))
        angle = position[:, None] * freq[None, :]
        cos_vals = tl.cos(angle)
        sin_vals = tl.sin(angle)
        
        acc_real = acc * cos_vals
        acc_imag = tl.roll(acc, 1, axis=1) * sin_vals
        acc = acc_real + acc_imag
    
    out_ptrs = output_ptr + m_offset[:, None] * stride_outm + n_offset[None, :] * stride_outn
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

def rms_matmul_rbe_wrapper(x: torch.Tensor, w: torch.Tensor, rms_w: torch.Tensor, 
                          apply_rotary: bool = False, theta: float = 10000.0):
    assert x.is_cuda and w.is_cuda and rms_w.is_cuda
    assert x.dim() >= 2 and w.dim() == 2
    *batch_dims, M, K = x.shape
    N = w.size(1)
    
    x_flat = x.view(-1, K)
    output = torch.empty((*batch_dims, M, N), device=x.device, dtype=x.dtype)
    output_flat = output.view(-1, N)
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 64
    
    grid = (
        triton.cdiv(x_flat.size(0), BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    rms_matmul_rbe[grid](
        x_flat, w, rms_w, output_flat,
        x_flat.size(0), N, K,
        x_flat.stride(0), x_flat.stride(1),
        w.stride(0), w.stride(1),
        output_flat.stride(0), output_flat.stride(1),
        THETA=theta,
        apply_rotary=apply_rotary,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        eps=1e-6
    )
    
    return output
