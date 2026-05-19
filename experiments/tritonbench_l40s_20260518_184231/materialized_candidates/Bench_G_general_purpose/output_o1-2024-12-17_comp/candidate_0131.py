import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Loop over K dimension
    for k_block_start in range(0, K, BLOCK_SIZE_K):
        k_mask = offs_k + k_block_start < K

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am
                          + (offs_k[None, :] + k_block_start) * stride_ak)
        b_ptrs = B_ptr + ((offs_k[:, None] + k_block_start) * stride_bk
                          + offs_n[None, :] * stride_bn)

        a = tl.where(mask_m[:, None] & k_mask[None, :], tl.load(a_ptrs), 0.0)
        b = tl.where(k_mask[:, None] & mask_n[None, :], tl.load(b_ptrs), 0.0)
        acc += tl.dot(a, b)

    if ACTIVATION == 1:
        # Leaky ReLU
        acc = tl.where(acc >= 0, acc, acc * 0.01)

    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=mask)


def matmul(A, B, activation=None):
    assert A.is_contiguous(), "A must be contiguous"
    assert B.is_contiguous(), "B must be contiguous"
    assert A.dtype == B.dtype, "A and B must have the same dtype"
    M, K = A.shape
    K_b, N = B.shape
    assert K == K_b, "Inner dimensions must match"
    ACTIVATION = 1 if activation == "leaky_relu" else 0

    import math
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    grid = (
        math.ceil(M / BLOCK_SIZE_M),
        math.ceil(N / BLOCK_SIZE_N)
    )

    C = A.new_empty((M, N))
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        ACTIVATION=ACTIVATION
    )
    return C
