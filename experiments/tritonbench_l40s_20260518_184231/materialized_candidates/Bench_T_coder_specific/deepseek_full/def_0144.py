import torch
import triton
import triton.language as tl

@triton.jit
def kernel(A, B, alpha, beta, C, M, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    p2m = tl.cdiv(M, BLOCK_SIZE)
    p2n = tl.cdiv(N, BLOCK_SIZE)
    pidm = pid // p2n
    pidn = pid % p2n
    if pidm * p2n + pidn >= p2m * p2n or pidn >= p2n:
        return
    offs_am = pidm * BLOCK_SIZE
    offs_bn = pidn * BLOCK_SIZE
    a_ptrs = A + offs_am * K + tl.arange(0, BLOCK_SIZE)[:, None] * K + tl.arange(0, BLOCK_SIZE)[None, :]
    b_ptrs = B + offs_bn * N + tl.arange(0, BLOCK_SIZE)[:, None] * N + tl.arange(0, BLOCK_SIZE)[None, :]
    c_ptrs = C + offs_am * N + tl.arange(0, BLOCK_SIZE)[:, None] * N + tl.arange(0, BLOCK_SIZE)[None, :]
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE):
        k_off = k + tl.arange(0, BLOCK_SIZE)[:, None] * K + tl.arange(0, BLOCK_SIZE)[None, :]
        a = tl.load(a_ptrs + k_off, mask=k_off < K * K, other=0.0)
        b = tl.load(b_ptrs + k_off, mask=k_off < K * N, other=0.0)
        acc += tl.dot(a, b)
    acc = acc * alpha
    c_old = tl.load(c_ptrs, mask=c_ptrs < M * N, other=0.0)
    acc = acc + c_old * beta
    tl.store(c_ptrs, acc, mask=c_ptrs < M * N)

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    _, N = B.shape
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE']) * triton.cdiv(N, META['BLOCK_SIZE']),)
    kernel[grid](A, B, alpha, beta, C, M, N, K)
    return torch.dot(C[0], C[1])
