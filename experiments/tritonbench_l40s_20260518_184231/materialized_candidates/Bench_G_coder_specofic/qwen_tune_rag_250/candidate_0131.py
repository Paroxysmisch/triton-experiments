import triton
import triton.language as tl
import kernel_utils

@triton.jit
def kernel(C, A, B, M, N, K,
           stride_cm, stride_cn,
           stride_am, stride_ak,
           stride_bk, stride_bn,
           BLOCK_M: tl.constexpr,
           BLOCK_N: tl.constexpr,
           BLOCK_K: tl.constexpr):
    # Triton kernel for matrix multiplication with extra elementwise operation
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = kernel_utils.mul(accumulator, accumulator)
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)


def matmul(A, B, activation=""):
    # Function to invoke the Triton kernel for matrix multiplication
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    if activation == "leaky_relu":
        ACTIVATION = kernel_utils.leaky_relu
    else:
        ACTIVATION = None

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    kernel[grid](
        C,
        A,
        B,
        M,
        N,
        K,
        C.stride(0),
        C.stride(1),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        ACTIVATION,
    )
    return C
