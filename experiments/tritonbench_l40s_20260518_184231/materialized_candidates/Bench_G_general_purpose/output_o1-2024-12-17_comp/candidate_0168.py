import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_stages": 2, "num_warps": 4}),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_stages": 2, "num_warps": 8}),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "num_stages": 2, "num_warps": 8}),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "num_stages": 3, "num_warps": 8}),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mm = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        a_ptrs = A + (offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak)
        b_ptrs = B + ((k_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_start + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k_start + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        mm += tl.dot(a, b)

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, mm, mask=mask)


def triton_matmul(a, b):
    import torch
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "Incompatible dimensions"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (( (M + META['BLOCK_SIZE_M'] - 1) // META['BLOCK_SIZE_M'] ) * ( (N + META['BLOCK_SIZE_N'] - 1) // META['BLOCK_SIZE_N'] ),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
