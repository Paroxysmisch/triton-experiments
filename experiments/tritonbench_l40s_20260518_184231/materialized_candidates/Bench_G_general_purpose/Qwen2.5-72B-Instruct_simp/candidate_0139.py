import triton
import triton.language as tl

# Configuration class to hold kernel execution parameters
class Config:
    def __init__(self, num_warps=4, num_stages=3, num_ctas=1):
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_ctas = num_ctas
        self.kwargs = {
            'num_warps': num_warps,
            'num_stages': num_stages,
            'num_ctas': num_ctas
        }

# Kernel for integer matrix multiplication C = A x B
@triton.jit
def matmul_kernel_with_block_pointers(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = tl.arange(0, BLOCK_M) + (pid % grid_m) * BLOCK_M
    pid_n = tl.arange(0, BLOCK_N) + (pid // grid_m) * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + pid_m[:, None] * stride_am + (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak, mask=k + tl.arange(0, BLOCK_K) < K, other=0)
        b = tl.load(B_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * stride_bk + pid_n[None, :] * stride_bn, mask=k + tl.arange(0, BLOCK_K) < K, other=0)
        acc += tl.dot(a, b)

    tl.store(C_ptr + pid_m[:, None] * stride_cm + pid_n[None, :] * stride_cn, acc, mask=(pid_m[:, None] < M) & (pid_n[None, :] < N))

# Kernel for scaled integer matrix multiplication C = A x B * scales
@triton.jit
def scaled_matmul_kernel_with_block_pointers(
    A_ptr, B_ptr, C_ptr, scales_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_sm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = tl.arange(0, BLOCK_M) + (pid % grid_m) * BLOCK_M
    pid_n = tl.arange(0, BLOCK_N) + (pid // grid_m) * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + pid_m[:, None] * stride_am + (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak, mask=k + tl.arange(0, BLOCK_K) < K, other=0)
        b = tl.load(B_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * stride_bk + pid_n[None, :] * stride_bn, mask=k + tl.arange(0, BLOCK_K) < K, other=0)
        acc += tl.dot(a, b)

    scales = tl.load(scales_ptr + pid_m * stride_sm, mask=pid_m < M, other=1)
    acc = acc * scales[:, None]

    tl.store(C_ptr + pid_m[:, None] * stride_cm + pid_n[None, :] * stride_cn, acc, mask=(pid_m[:, None] < M) & (pid_n[None, :] < N))

# Wrapper function for integer matrix multiplication
def int_matmul_kernel(A, B, C, M, N, K, config: Config):
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel_with_block_pointers[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=config.kwargs['num_warps'] * 32,
        BLOCK_N=config.kwargs['num_warps'] * 32,
        BLOCK_K=32,
        **config.kwargs
    )

# Wrapper function for scaled integer matrix multiplication
def int_scaled_matmul_kernel(A, B, C, scales, M, N, K, config: Config):
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    scaled_matmul_kernel_with_block_pointers[grid](
        A, B, C, scales,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        scales.stride(0),
        BLOCK_M=config.kwargs['num_warps'] * 32,
        BLOCK_N=config.kwargs['num_warps'] * 32,
        BLOCK_K=32,
        **config.kwargs
    )
