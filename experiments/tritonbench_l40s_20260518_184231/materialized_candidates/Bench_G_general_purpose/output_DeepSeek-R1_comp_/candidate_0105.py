import torch
import triton
import triton.language as tl

@triton.jit
def rbe_triton(
    x_ptr, out_ptr,
    stride_batch, stride_m, stride_k,
    K_total, M_total,
    theta: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    offs_batch = pid_batch
    offs_m = pid_m * BLOCK_SIZE_M
    offs_k = pid_k * BLOCK_SIZE_K
    
    BLOCK_SIZE_K_COMPLEX = BLOCK_SIZE_K // 2
    i_in_block = tl.arange(0, BLOCK_SIZE_K_COMPLEX)
    m_in_block = tl.arange(0, BLOCK_SIZE_M)
    
    i = (offs_k // 2) + i_in_block
    m_global = offs_m + m_in_block[:, None]
    
    d = K_total // 2
    freq_i = 1.0 / (theta ** ((2.0 * i.to(tl.float32)) / d))
    angle = m_global.to(tl.float32) * freq_i
    
    cos_theta = tl.cos(angle)
    sin_theta = tl.sin(angle)
    
    k_real = offs_k + 2 * i_in_block
    k_imag = k_real + 1
    
    mask_real = (m_global < M_total) & (k_real < K_total)
    mask_imag = (m_global < M_total) & (k_imag < K_total)
    
    x_real_ptr = x_ptr + offs_batch * stride_batch + m_global * stride_m + k_real * stride_k
    x_imag_ptr = x_ptr + offs_batch * stride_batch + m_global * stride_m + k_imag * stride_k
    
    real = tl.load(x_real_ptr, mask=mask_real, other=0.0)
    imag = tl.load(x_imag_ptr, mask=mask_imag, other=0.0)
    
    out_real = real * cos_theta - imag * sin_theta
    out_imag = real * sin_theta + imag * cos_theta
    
    out_real_ptr = out_ptr + offs_batch * stride_batch + m_global * stride_m + k_real * stride_k
    out_imag_ptr = out_ptr + offs_batch * stride_batch + m_global * stride_m + k_imag * stride_k
    
    tl.store(out_real_ptr, out_real, mask=mask_real)
    tl.store(out_imag_ptr, out_imag, mask=mask_imag)

def rbe_triton_wrapper(x: torch.Tensor) -> torch.Tensor:
    batch, M, K = x.shape
    out = torch.empty_like(x)
    assert x.is_cuda and out.is_cuda, "Input and output must be on GPU"
    BLOCK_SIZE_M = 2
    BLOCK_SIZE_K = 1024
    grid = (batch, triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_K))
    theta = 10000.0
    rbe_triton[grid](
        x, out,
        x.stride(0), x.stride(1), x.stride(2),
        K, M,
        theta=theta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return out
