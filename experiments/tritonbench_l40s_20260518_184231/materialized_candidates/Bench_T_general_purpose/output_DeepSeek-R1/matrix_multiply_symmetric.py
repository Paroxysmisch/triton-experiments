import torch
import triton
import triton.language as tl

@triton.jit
def fused_matmul_add_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    M, N, K,
    stride_a_row, stride_a_col,
    stride_b_row, stride_b_col,
    stride_c_row, stride_c_col,
    stride_d_row, stride_d_col,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_a_row + offs_k[None, :] * stride_a_col
    b_ptrs = b_ptr + offs_k[:, None] * stride_b_row + offs_bn[None, :] * stride_b_col

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k), other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_a_col
        b_ptrs += BLOCK_SIZE_K * stride_b_row

    accumulator *= alpha

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_c_row + offs_cn[None, :] * stride_c_col
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c = tl.load(c_ptrs, mask=c_mask, other=0.0)
    c *= beta

    d = accumulator + c

    offs_dm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_dn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    d_ptrs = d_ptr + offs_dm[:, None] * stride_d_row + offs_dn[None, :] * stride_d_col
    d_mask = (offs_dm[:, None] < M) & (offs_dn[None, :] < N)
    tl.store(d_ptrs, d, mask=d_mask)

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n, m = A.shape
    m_b, p = B.shape
    assert m == m_b and C.shape == (n, p), "Input dimensions mismatch"
    C_new = torch.empty_like(C)

    def grid_step1(META):
        return (triton.cdiv(n, META['BLOCK_SIZE_M']) * triton.cdiv(p, META['BLOCK_SIZE_N']), )

    fused_matmul_add_kernel[grid_step1](
        A, B, C, C_new,
        n, p, m,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        C_new.stride(0), C_new.stride(1),
        alpha,
        beta,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=64,
    )

    assert p == n, "C must be square after first step for the second operation"
    C_final = torch.empty(n, n, dtype=C.dtype, device=C.device)

    def grid_step2(META):
        return (triton.cdiv(n, META['BLOCK_SIZE_M']) * triton.cdiv(n, META['BLOCK_SIZE_N']), )

    fused_matmul_add_kernel[grid_step2](
        C_new, C_new.T, C_new, C_final,
        n, n, p,
        C_new.stride(0), C_new.stride(1),
        C_new.T.stride(0), C_new.T.stride(1),
        C_new.stride(0), C_new.stride(1),
        C_final.stride(0), C_final.stride(1),
        alpha,
        beta,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=64,
    )

    return C_final
