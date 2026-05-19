import torch
import triton
import triton.language as tl

# Triton kernel
@triton.jit
def matmul_scale_kernel(
    A_ptr, B_ptr, C_ptr, alpha, beta, n, m, p, stride_am, stride_ap, stride_bm, stride_bp, stride_cn, stride_cp,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_n = (n + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_p = (p + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    n_block = pid // grid_p
    p_block = pid % grid_p

    offs_n = n_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_p = p_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    A_ptrs = A_ptr + (offs_n[:, None] * stride_am + offs_k[None, :] * stride_ap)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bm + offs_p[None, :] * stride_bp)
    C_ptrs = C_ptr + (offs_n[:, None] * stride_cn + offs_p[None, :] * stride_cp)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, m, BLOCK_SIZE_K):
        a = tl.load(A_ptrs, mask=(offs_n[:, None] < n) & (offs_k[None, :] < m), other=0.0)
        b = tl.load(B_ptrs, mask=(offs_k[:, None] < m) & (offs_p[None, :] < p), other=0.0)
        acc += tl.dot(a, b)
        A_ptrs += BLOCK_SIZE_K * stride_ap
        B_ptrs += BLOCK_SIZE_K * stride_bm

    c = tl.load(C_ptrs, mask=(offs_n[:, None] < n) & (offs_p[None, :] < p), other=0.0)
    acc = alpha * acc + beta * c
    tl.store(C_ptrs, acc, mask=(offs_n[:, None] < n) & (offs_p[None] < p))

# Wrapper function
def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    assert A.shape[1] == B.shape[0], "Inner dimensions must match for matrix multiplication"
    assert C.shape[0] >= 2, "Matrix C must have at least two rows for the dot product"

    n, m = A.shape
    _, p = B.shape

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    grid = ((n + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * ((p + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)

    matmul_scale_kernel[grid](
        A_ptr=A,
        B_ptr=B,
        C_ptr=C,
        alpha=alpha,
        beta=beta,
        n=n,
        m=m,
        p=p,
        stride_am=A.stride(0),
        stride_ap=A.stride(1),
        stride_bm=B.stride(0),
        stride_bp=B.stride(1),
        stride_cn=C.stride(0),
        stride_cp=C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    # Compute the dot product of the first two rows of the updated C
    result = torch.dot(C[0], C[1])
    return result
