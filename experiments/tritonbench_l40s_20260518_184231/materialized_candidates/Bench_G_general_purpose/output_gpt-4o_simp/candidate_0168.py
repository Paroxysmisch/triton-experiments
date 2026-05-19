import triton
import triton.language as tl

# Define the matrix multiplication kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 2}),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    group_id = pid // GROUP_SIZE_M
    within_group_id = pid % GROUP_SIZE_M

    block_m = group_id * GROUP_SIZE_M + within_group_id
    block_n = tl.program_id(1)

    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    A_block_ptr = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_block_ptr = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        A_block = tl.load(A_block_ptr, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        B_block = tl.load(B_block_ptr, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(A_block, B_block)

        A_block_ptr += BLOCK_SIZE_K * stride_ak
        B_block_ptr += BLOCK_SIZE_K * stride_bk

    C_block_ptr = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_block_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# Wrapper function to facilitate calling the kernel
def triton_matmul(A, B):
    assert A.shape[1] == B.shape[0], "Incompatible dimensions for matrix multiplication"
    M, K = A.shape
    _, N = B.shape

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * META['GROUP_SIZE_M'],
                         triton.cdiv(N, META['BLOCK_SIZE_N']))

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )

    return C
