import triton
import triton.language as tl

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
    # Calculate the program IDs along the two major axes
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the starting position of the thread block in the output matrix
    rm = pid_m * BLOCK_SIZE_M
    rn = pid_n * BLOCK_SIZE_N

    # Initialize the accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Calculate the offsets for the sub-blocks of a and b
        offs_am = rm + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = rn + tl.arange(0, BLOCK_SIZE_N)
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)

        # Load the sub-blocks of a and b
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=offs_k[None, :] < K, other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn, mask=offs_k[:, None] < K, other=0.0)

        # Perform the dot product and accumulate the result
        accumulator += tl.dot(a, b)

    # Store the result in the output matrix c
    c = accumulator.to(tl.float16)
    offs_cm = rm + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = rn + tl.arange(0, BLOCK_SIZE_N)
    c_ptr += offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptr, c, mask=offs_cm[:, None] < M & offs_cn[None, :] < N)

import torch
import triton
import triton.language as tl

def matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    # Ensure the input tensors are on the same device
    assert a.device == b.device == c.device
    assert a.dtype == b.dtype == torch.float16
    assert c.dtype == torch.float16

    # Get the dimensions of the matrices
    M, K = a.shape
    K, N = b.shape

    # Define the block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16

    # Calculate the grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), 1)

    # Launch the kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

# Example usage
M, K, N = 1024, 1024, 1024
a = torch.randn((M, K), device='cuda', dtype=torch.float16)
b = torch.randn((K, N), device='cuda', dtype=torch.float16)
c = torch.empty((M, N), device='cuda', dtype=torch.float16)

matmul(a, b, c)

print(c)
