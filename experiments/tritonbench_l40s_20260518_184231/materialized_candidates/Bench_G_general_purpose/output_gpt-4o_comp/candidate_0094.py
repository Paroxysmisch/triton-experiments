import triton
import triton.language as tl

# Define the Triton kernel for matrix multiplication
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program ID for each block along the M and N dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate the starting position of each block in the output matrix
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator for the block
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the K dimension in blocks
    for k_block in range(0, K, BLOCK_SIZE_K):
        # Load sub-blocks of A and B
        a_block = tl.load(a_ptr + offs_am[:, None] * stride_am + (k_block + offs_k)[None, :] * stride_ak, mask=offs_am[:, None] < M)
        b_block = tl.load(b_ptr + (k_block + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn, mask=offs_bn[None, :] < N)

        # Perform dot product and accumulate
        accumulator += tl.dot(a_block, b_block)

    # Store the result in C, casting to float16
    c_result = accumulator.to(tl.float16)
    c_ptr = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_ptr, c_result, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

# Define the wrapper function to launch the kernel
def matmul(a, b, c, M, N, K, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32):
    # Define grid sizes based on the matrix dimensions and block sizes
    grid = (
        (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    )

    # Calculate strides for accessing elements in memory
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)

    # Launch the kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
