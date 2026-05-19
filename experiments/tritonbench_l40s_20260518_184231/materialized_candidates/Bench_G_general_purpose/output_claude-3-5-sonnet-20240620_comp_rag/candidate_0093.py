import triton
import triton.language as tl
import kernel_utils

@triton.jit
def matmul_kernel(
    C, A, B, M, N, K,
    stride_cm, stride_cn,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Compute program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Compute pointers
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load blocks from A and B
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        
        # Perform matrix multiplication
        accumulator += tl.dot(a, b)
        
        # Move pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Perform elementwise multiplication: C = (A x B) * (A x B)
    c = kernel_utils.mul(accumulator, accumulator)

    # Write output
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)

# Wrapper function
def matmul(C, A, B, M, N, K, block_size_m=32, block_size_n=32, block_size_k=32):
    # Define grid
    grid = (triton.cdiv(M, block_size_m), triton.cdiv(N, block_size_n))
    
    # Define strides
    stride_cm, stride_cn = C.stride(0), C.stride(1)
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)

    # Launch kernel
    matmul_kernel[grid](
        C, A, B, M, N, K,
        stride_cm, stride_cn,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        BLOCK_M=block_size_m,
        BLOCK_N=block_size_n,
        BLOCK_K=block_size_k
    )
