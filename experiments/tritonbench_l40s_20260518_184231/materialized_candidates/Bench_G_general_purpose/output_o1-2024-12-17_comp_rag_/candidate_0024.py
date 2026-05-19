import triton
import triton.language as tl
import torch

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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for each block of rows/columns
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Accumulator for partial sums
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k_block in range(0, K, BLOCK_SIZE_K):
        # Load sub-block of a
        a_offsets = (offs_m[:, None] * stride_am) + ((k_block + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_ak)
        a = tl.load(a_ptr + a_offsets, mask=(offs_m[:, None] < M) & ((k_block + tl.arange(0, BLOCK_SIZE_K))[None, :] < K), other=0.0)

        # Load sub-block of b
        b_offsets = ((k_block + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)
        b = tl.load(b_ptr + b_offsets, mask=((k_block + tl.arange(0, BLOCK_SIZE_K))[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Matmul accumulation
        acc += tl.dot(a, b)

    # Write back to C
    c_offsets = (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + c_offsets, acc, mask=mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matmul."
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # Create output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Define block sizes and grid
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    grid = (
        (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    )

    # Convert strides to ints
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    # Launch Triton kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_stages=3,
        num_warps=4
    )
    return c
