import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_update_kernel(A, out, M, N, K, b_k: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr):
    k_block_idx = tl.program_id(0)
    n_block_idx = tl.program_id(1)
    m_block_idx = tl.program_id(2)

    offs_k = tl.arange(0, b_k)
    offs_n = n_block_idx * K + tl.arange(0, b_k)
    offs_m = m_block_idx * K + tl.arange(0, b_k)

    a_ptrs = A + offs_m[:, None] * N * K + offs_k[None, :] * K + offs_k[:, None]
    acc = tl.zeros((b_k, b_k), dtype=tl.float32)
    mask = (offs_k[None, :] < M) & (offs_k[:, None] < M)

    for _ in range(K // b_k):
        a = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.dot(a, a.trans(1))
        a_ptrs += b_k
    acc *= alpha
    acc = acc.to(A.dtype.element_ty)

    c_ptrs = out + offs_m[:, None] * N * K + offs_n[None, :]
    d_ptrs = out + offs_n[:, None] * N * K + offs_m[None, :]

    b = tl.load(c_ptrs, mask=mask, other=0.0)
    b = b.to(tl.float32)
    c = acc + beta * b
    tl.store(d_ptrs, c.to(A.dtype.element_ty), mask=(offs_k[:, None] < M) & (offs_k[None, :] < N))


def matrix_multiply_symmetric(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float
) -> torch.Tensor:
    assert A.shape[0] == A.shape[1], "Shape must be same"
    assert B.shape[0] == B.shape[1], "Shape must be same"
    assert A.shape[0] == B.shape[0], "Shape must be same"

    assert C.shape[0] == C.shape[1], "Shape must be same"
    assert A.shape[0] == C.shape[0], "Shape must be same"

    out = C

    M = N = K = A.shape[0]
    grid = lambda META: (
        triton.cdiv(M, META["b_k"]),
        triton.cdiv(N, META["b_k"]),
        triton.cdiv(K, META["b_k"])
    )
    _symmetric_update_kernel[grid](
        A, out, M, N, K,
        B.shape[0],
        alpha, beta
    )

    return out
