import triton
import triton.language as tl

# Configuration class to hold kernel execution parameters
class Config:
    def __init__(self, num_warps=4, num_stages=3, num_ctas=1):
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
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Compute block indices
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    block_m = pid // grid_n
    block_n = pid % grid_n

    # Initialize pointers for the current block
    a_ptr = A_ptr + block_m * BLOCK_M * stride_am
    b_ptr = B_ptr + block_n * BLOCK_N * stride_bn
    c_ptr = C_ptr + block_m * BLOCK_M * stride_cm + block_n * BLOCK_N

    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load A and B blocks
        a_block = tl.load(a_ptr + k * stride_ak, mask=(k + tl.arange(0, BLOCK_K)[:, None] < K))
        b_block = tl.load(b_ptr + k * stride_bk, mask=(k + tl.arange(0, BLOCK_K) < K))

        # Matrix multiplication
        acc += tl.dot(a_block, b_block)

    # Store result
    c_mask = (block_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < M) & (block_n * BLOCK_N + tl.arange(0, BLOCK_N) < N)
    tl.store(c_ptr, acc, mask=c_mask)

# Kernel for scaled integer matrix multiplication C = A x B * scales1
@triton.jit
def scaled_matmul_kernel_with_block_pointers(
    A_ptr, B_ptr, C_ptr, scales1_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Compute block indices
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    block_m = pid // grid_n
    block_n = pid % grid_n

    # Initialize pointers for the current block
    a_ptr = A_ptr + block_m * BLOCK_M * stride_am
    b_ptr = B_ptr + block_n * BLOCK_N * stride_bn
    c_ptr = C_ptr + block_m * BLOCK_M * stride_cm + block_n * BLOCK_N
    scales1_ptr = scales1_ptr + block_m * BLOCK_M

    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load A and B blocks
        a_block = tl.load(a_ptr + k * stride_ak, mask=(k + tl.arange(0, BLOCK_K)[:, None] < K))
        b_block = tl.load(b_ptr + k * stride_bk, mask=(k + tl.arange(0, BLOCK_K) < K))

        # Matrix multiplication
        acc += tl.dot(a_block, b_block)

    # Load scales and apply scaling
    scales1 = tl.load(scales1_ptr)
    acc *= scales1[:, None]

    # Store result
    c_mask = (block_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < M) & (block_n * BLOCK_N + tl.arange(0, BLOCK_N) < N)
    tl.store(c_ptr, acc, mask=c_mask)

# Wrapper function for integer matrix multiplication
def int_matmul_kernel(A, B, C, M, N, K, config):
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel_with_block_pointers[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
        **config.kwargs
    )

# Wrapper function for scaled integer matrix multiplication
def int_scaled_matmul_kernel(A, B, C, scales1, M, N, K, config):
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    scaled_matmul_kernel_with_block_pointers[grid](
        A, B, C, scales1, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
        **config.kwargs
    )
