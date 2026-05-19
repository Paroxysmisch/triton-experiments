import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_matmul_and_abs_sum_kernel(
    A,
    C,
    M,
    N,
    lda,
    ldc,
    alpha,
    beta,
    PLACEHOLDER_1,
    PLACEHOLDER_2,
    PLACEHOLDER_3,
    PLACEHOLDER_4,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    A = A + (m_offset * lda + n_offset)
    C = C + (m_offset * ldc + n_offset)
    sum_ = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        a = tl.load(A, mask=(m_offset < M) & (n_offset < N), other=0.0)
        c = tl.load(C, mask=(m_offset < M) & (n_offset < N), other=0.0)
        a = tl.trans(a)
        c = tl.trans(c)
        sum_ += alpha * tl.dot(a, a, allow_tf32=False) + beta * c
        A += BLOCK_SIZE_N
        C += BLOCK_SIZE_N
    sum_ = tl.where((m_offset < M) & (n_offset < N), sum_, 0.0)
    tl.store(C, sum_, mask=(m_offset < M) & (n_offset < N))


def triton_symmetric_mm_and_abs_sum(A, C, alpha, beta):
    trans_a = torch.t(A)
    out = torch.mm(trans_a, A)
    out *= alpha
    out = out.to(C.dtype)
    orig_c = C.clone()
    C *= beta
    M, N = out.shape
    asum = torch.abs(out + C).sum()
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _symmetric_matmul_and_abs_sum_kernel[grid](
        A,
        C,
        M,
        N,
        trans_a.stride(0),
        C.stride(0),
        alpha,
        beta,
        orig_c,
        out,
        asum,
        trans_a,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
    )
    asum += torch.abs(C - orig_c).sum()
    return asum
