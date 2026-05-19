import triton
import triton.language as tl
import torch

# Triton kernel for matrix multiplication
@triton.jit
def matmul_kernel(a, b, c, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    stride_am = a.strides[0]
    stride_ak = a.strides[1]
    stride_bk = b.strides[0]
    stride_bn = b.strides[1]
    stride_cm = c.strides[0]
    stride_cn = c.strides[1]
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_am
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_bn
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a + offs_am[:, None] * stride_ak + offs_k[None, :] * stride_ak
    b_ptrs = b + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a_block = tl.load(a_ptrs)
        b_block = tl.load(b_ptrs)
        accumulator += tl.dot(a_block, b_block)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c_ptrs = c + offs_am[:, None] * stride_cn + offs_bn[None, :] * stride_cn
    tl.store(c_ptrs, accumulator.to(c.dtype.element_ty))

# Function to call the Triton kernel
def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](a, b, c, M, N, K)
    return c

# Example usage
a = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
b = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
c = matmul(a, b)
