import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    x_ptr, y_ptr, z_ptr,
    # Matrix dimensions
    M, K, N,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # Strides
    stride_xm, stride_xk,
    stride_yk, stride_yn,
    stride_zm, stride_zn,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    y_ptrs = y_ptr + offs_k[:, None] * stride_yk + offs_n[None, :] * stride_yn
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        _BLOCK_K = min(BLOCK_K, k_remaining)

        a = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(y_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        
        accumulator += tl.dot(a, b)
        
        x_ptrs += BLOCK_K * stride_xk
        y_ptrs += BLOCK_K * stride_yk

    offs_zm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_zn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    z_ptrs = z_ptr + offs_zm[:, None] * stride_zm + offs_zn[None, :] * stride_zn
    tl.store(z_ptrs, accumulator, mask=(offs_zm[:, None] < M) & (offs_zn[None, :] < N))

def matmul(x, y):
    M, K = x.shape
    K, N = y.shape
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    
    matmul_kernel[grid](
        x, y, z,
        M, K, N,
        BLOCK_M, BLOCK_N, BLOCK_K,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        z.stride(0), z.stride(1),
    )
    return z
