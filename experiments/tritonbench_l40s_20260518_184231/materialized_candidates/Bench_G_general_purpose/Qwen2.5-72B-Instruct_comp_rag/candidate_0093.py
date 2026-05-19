import triton
import triton.language as tl

# Utility function for element-wise multiplication
def mul(a, b):
    return a * b

# Triton kernel for matrix multiplication with extra elementwise operation
@triton.jit
def matmul_kernel(C, A, B, M, N, K,
                  stride_cm, stride_cn,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    # Calculate the program IDs for the current thread block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate the offsets for the current thread block
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Calculate the pointers for the current block of A and B
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize the accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # Perform the dot product and accumulate the result
        accumulator += tl.dot(a, b)
        # Advance the pointers to the next K block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Perform the element-wise multiplication
    c = mul(accumulator, accumulator)

    # Calculate the offsets for the output matrix C
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    # Store the result back to the output matrix C
    tl.store(c_ptrs, c)

# Wrapper function to configure the kernel launch
def matmul(A, B, C, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    # Calculate the grid size
    grid = (tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N), 1)

    # Launch the kernel
    matmul_kernel[grid](C, A, B, M, N, K,
                        C.stride(0), C.stride(1),
                        A.stride(0), A.stride(1),
                        B.stride(0), B.stride(1),
                        BLOCK_M, BLOCK_N, BLOCK_K)
