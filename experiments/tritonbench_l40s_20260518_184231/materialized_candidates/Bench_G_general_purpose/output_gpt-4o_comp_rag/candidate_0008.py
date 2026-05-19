import triton
import triton.language as tl
import kernel_utils

@triton.jit
def matmul_square_kernel(C, A, B, M, N, K,
                         stride_cm, stride_cn,
                         stride_am, stride_ak,
                         stride_bk, stride_bn,
                         BLOCK_M: tl.constexpr,
                         BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr):
    # Triton kernel for matrix multiplication followed by elementwise squaring
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate offsets for A and B matrices
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the blocks of A and B
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Block-wise matrix multiplication
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load blocks of A and B, apply mask for boundary conditions
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # Accumulate the dot product
        accumulator += tl.dot(a, b)
        # Advance pointers to the next block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Elementwise square of the accumulated result
    c = kernel_utils.mul(accumulator, accumulator)

    # Store the result in C
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)

def matmul_square(a, b):
    # Assume a is (M, K) and b is (K, N)
    M, K = a.shape
    _, N = b.shape

    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Create output tensor
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)

    # Calculate strides
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    # Launch kernel
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    matmul_square_kernel[grid](
        c, a, b, M, N, K,
        stride_cm, stride_cn,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    return c
