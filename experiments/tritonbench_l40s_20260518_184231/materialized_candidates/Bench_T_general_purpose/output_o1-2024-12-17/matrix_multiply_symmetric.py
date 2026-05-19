import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_mm(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ALPHA, BETA,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # Create pointers for A and B
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Loop over K dimension
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    # Scale by alpha
    acc = acc * ALPHA
    # Add beta*C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_old = tl.load(c_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None] < N), other=0.0)
    acc += c_old * BETA
    # Write back
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None] < N))

def _gemm_triton(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float):
    # Shapes
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, "Incompatible dimensions for matrix multiplication."
    # Grid
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    
    triton.run(
        _kernel_mm,
        grid=grid,
        num_warps=4,
        num_stages=2,
        args=[
            A.data_ptr(), B.data_ptr(), C.data_ptr(),
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            alpha, beta
        ],
        kwargs={
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K
        }
    )

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # First operation: C = alpha * mm(A, B) + beta * C
    _gemm_triton(A, B, C, alpha, beta)
    # Second operation: C = alpha * mm(C, C.T) + beta * C
    # We need a temp to pass the transpose of C to the kernel
    Ct = C.t().contiguous()
    _gemm_triton(C, Ct, C, alpha, beta)
    return C
