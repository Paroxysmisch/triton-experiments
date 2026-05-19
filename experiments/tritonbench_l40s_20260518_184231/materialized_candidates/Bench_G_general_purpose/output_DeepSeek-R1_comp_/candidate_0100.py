import triton
import triton.language as tl
import torch

def leaky_relu(x, alpha):
    return tl.where(x >= 0, x, alpha * x)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    apply_leaky: tl.constexpr,
    alpha: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    block_m = pid_m * BLOCK_SIZE_M
    block_n = pid_n * BLOCK_SIZE_N
    
    off_m = block_m + tl.arange(0, BLOCK_SIZE_M)
    off_n = block_n + tl.arange(0, BLOCK_SIZE_N)
    
    a_ptrs = a_ptr + off_m[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak
    b_ptrs = b_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk + off_n[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + tl.arange(0, BLOCK_SIZE_K)[:, None] < K), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    if apply_leaky:
        accumulator = leaky_relu(accumulator, alpha)
    
    c_ptrs = c_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
    mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)

def matmul(a, b, apply_leaky=False, alpha=0.1):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        apply_leaky=apply_leaky,
        alpha=alpha,
    )
    
    return c

# Example usage
a = torch.randn(1024, 512, device='cuda')
b = torch.randn(512, 2048, device='cuda')

# Regular matrix multiplication
c = matmul(a, b)

# With leaky ReLU activation
c_with_activation = matmul(a, b, apply_leaky=True, alpha=0.01)
