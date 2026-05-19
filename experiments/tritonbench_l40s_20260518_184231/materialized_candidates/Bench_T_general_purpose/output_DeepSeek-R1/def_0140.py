import torch
import triton
import triton.language as tl

@triton.jit
def tril_mm_kernel(
    A_ptr, B_ptr, C_ptr,
    n, p,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha_beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, n, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] >= (k + offs_k[None, :])) & (offs_m[:, None] < n), other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < n) & (offs_n[None, :] < p), other=0.0)
        
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    acc = acc * alpha_beta
    
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < n) & (offs_n[None, :] < p))

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
    n = A.size(0)
    p = B.size(1)
    assert A.size(1) == B.size(0), "A and B must have compatible dimensions for matrix multiplication"
    
    C = torch.empty((n, p), device=A.device, dtype=A.dtype)
    alpha_beta = alpha * beta
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    grid = (triton.cdiv(n, BLOCK_SIZE_M), triton.cdiv(p, BLOCK_SIZE_N))
    
    tril_mm_kernel[grid](
        A, B, C,
        n, p,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        alpha_beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return C
