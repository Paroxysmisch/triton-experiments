import torch
import triton
import triton.language as tl
import math

@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    TRANS_B: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    if TRANS_B:
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    else:
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk if not TRANS_B else BLOCK_SIZE_K * stride_bn
    
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c = tl.load(c_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    c = c * beta + accumulator * alpha
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.dim() == 2 and B.dim() == 2 and C.dim() == 2, "Inputs must be 2D matrices"
    n, m = A.shape
    m_, p = B.shape
    assert m == m_, "A and B matrices are not compatible for multiplication"
    assert C.shape == (n, p), "C matrix shape is incorrect"
    assert C.shape[0] == C.shape[1], "C must be square for the symmetric update"
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    C = C.clone()  # To avoid modifying the original C in-place
    
    grid_m = triton.cdiv(n, BLOCK_SIZE_M)
    grid_n = triton.cdiv(p, BLOCK_SIZE_N)
    gemm_kernel[(grid_m, grid_n)](
        A, B, C,
        M=n, N=p, K=m,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bk=B.stride(0), stride_bn=B.stride(1),
        stride_cm=C.stride(0), stride_cn=C.stride(1),
        alpha=alpha,
        beta=beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        TRANS_B=False,
    )
    
    grid_m = triton.cdiv(n, BLOCK_SIZE_M)
    grid_n = triton.cdiv(n, BLOCK_SIZE_N)
    gemm_kernel[(grid_m, grid_n)](
        C, C, C,
        M=n, N=n, K=n,
        stride_am=C.stride(0), stride_ak=C.stride(1),
        stride_bk=C.stride(1), stride_bn=C.stride(0),
        stride_cm=C.stride(0), stride_cn=C.stride(1),
        alpha=alpha,
        beta=beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        TRANS_B=True,
    )
    
    return C
