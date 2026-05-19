import triton
import triton.language as tl

@triton.jit
def batched_vecmat_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Compute the tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the starting index for each tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize the accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in blocks
    for k in range(0, K, BLOCK_K):
        # Load a block of A and B
        a = tl.load(A_ptr + (offs_m[:, None] * stride_am + (k + offs_k) * stride_ak))
        b = tl.load(B_ptr + ((k + offs_k)[:, None] * stride_bk + offs_n * stride_bn))

        # Compute the product and accumulate
        acc += tl.dot(a, b)

    # Store the result
    tl.store(C_ptr + (offs_m[:, None] * stride_cm + offs_n * stride_cn), acc)

def batched_vecmat(A, B, M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32):
    # Ensure the input matrices are contiguous
    A = A.contiguous()
    B = B.contiguous()

    # Allocate output matrix
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Compute strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Define grid size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch the kernel
    batched_vecmat_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    return C
