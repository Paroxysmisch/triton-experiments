import triton
import triton.language as tl


# Kernel for integer matrix multiplication: C = A x B
@triton.jit
def matmul_kernel_with_block_pointers(
    A_ptr, B_ptr, C_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # Compute block indices
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # Compute block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Block pointers
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator for the result
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        A = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0)
        B = tl.load(B_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0)
        accumulator += tl.dot(A, B)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Store the result
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Kernel for scaled integer matrix multiplication: C = (A x B) * scales1
@triton.jit
def scaled_matmul_kernel_with_block_pointers(
    A_ptr, B_ptr, scales1_ptr, C_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales1,
    M, N, K,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # Compute block indices
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # Compute block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Block pointers
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    scales1_ptrs = scales1_ptr + offs_m * stride_scales1

    # Accumulator for the result
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        A = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0)
        B = tl.load(B_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0)
        accumulator += tl.dot(A, B)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Apply scaling
    scales1 = tl.load(scales1_ptrs, mask=(offs_m < M), other=1).to(tl.int32)
    accumulator *= scales1[:, None]

    # Store the result
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Host-side wrapper for matmul_kernel_with_block_pointers
def int_matmul_kernel(a, b, c, config):
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = config.block_sizes
    grid = ((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N)

    matmul_kernel_with_block_pointers[grid](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )


# Host-side wrapper for scaled_matmul_kernel_with_block_pointers
def int_scaled_matmul_kernel(a, b, scales1, c, config):
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = config.block_sizes
    grid = ((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N)

    scaled_matmul_kernel_with_block_pointers[grid](
        a, b, scales1, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scales1.stride(0),
        M, N, K,
        EVEN_K=(K % BLOCK_K == 0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )


# Config class to encapsulate kernel configuration
class Config:
    def __init__(self, block_sizes=(128, 128, 32), num_warps=4, num_stages=2):
        self.block_sizes = block_sizes
        self.num_warps = num_warps
        self.num_stages = num_stages
