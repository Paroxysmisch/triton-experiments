import triton
import triton.language as tl

@triton.jit
def matmul_kernel(C, A, B, M, N, K,
                  stride_cm, stride_cn,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  BLOCK_SIZE_M: tl.constexpr, 
                  BLOCK_SIZE_N: tl.constexpr, 
                  BLOCK_SIZE_K: tl.constexpr):
    # Triton kernel for matrix multiplication with extra elementwise operation
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Load the next block of A and B
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_bn[None, :] < N, other=0.0)
        # Perform the dot product
        accumulator += tl.dot(a, b)
        # Advance the pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Perform the element-wise multiplication of the result with itself
    c = accumulator * accumulator

    # Store the result back to C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def matmul(C, A, B, M, N, K, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32):
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    matmul_kernel[grid](
        C, A, B, M, N, K,
        C.stride(0), C.stride(1),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
