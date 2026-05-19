import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Block starting offsets
    offs_m = pid_m * BLOCK_SIZE_M
    offs_n = pid_n * BLOCK_SIZE_N

    # Create a 2D range for indices
    offs_am = offs_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = offs_n + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension in steps of BLOCK_SIZE_K
    for k_block_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_block_start + tl.arange(0, BLOCK_SIZE_K)
        
        # Load sub-block of A
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        # Load sub-block of B
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk) + (offs_bn[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N), other=0.0)

        # Accumulate partial results
        accumulator += tl.dot(a, b)

    # Write back the result to C (cast to float16)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c = accumulator.to(tl.float16)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a_ptr, b_ptr, c_ptr, M, N, K,
           stride_am, stride_ak,
           stride_bk, stride_bn,
           stride_cm, stride_cn,
           BLOCK_SIZE_M=64, 
           BLOCK_SIZE_N=64, 
           BLOCK_SIZE_K=32):
    grid = (
        (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    )
    matmul_kernel[grid](
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
