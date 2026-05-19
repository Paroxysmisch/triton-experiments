import triton
import triton.language as tl
import math
import torch

@triton.jit
def _tril_mm_kernel(
    A_ptr, B_ptr, C_ptr,
    alpha, beta,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Initialize accumulator
    c = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b_ptrs = B_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Zero out elements above the diagonal of A
        row_ids = offs_m[:, None]
        col_ids = (k + offs_k[None, :])
        mask_tril = row_ids >= col_ids
        a = tl.where(mask_tril, a, 0.0)

        c += tl.dot(a, b)

    # Scale by alpha and beta
    c *= alpha * beta

    # Write back
    out_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(out_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2-D tensors."
    n, nA = A.shape
    nB, p = B.shape
    assert n == nA and n == nB, "Shapes must be (n,n) for A and (n,p) for B."

    # Create output
    C = torch.empty((n, p), device=A.device, dtype=A.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (math.ceil(n / BLOCK_M), math.ceil(p / BLOCK_N))

    _tril_mm_kernel[grid](
        A, B, C,
        alpha, beta,
        n, p, n,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    return C
