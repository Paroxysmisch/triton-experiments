import triton
import triton.language as tl
import torch

# Triton kernel for matrix multiplication with extra elementwise operation
@triton.jit
def matmul_kernel(C, A, B, M, N, K,
                  stride_cm, stride_cn,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    # Calculate the program ID and block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the offsets for the current block
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Compute the pointers for the current block
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
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the pointers to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Compute the elementwise square of the accumulator
    c = accumulator * accumulator

    # Write back the block of the output matrix C with masks
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)

# Wrapper function to launch the kernel
def matmul(A, B):
    # Check input dimensions
    M, K = A.shape
    K, N = B.shape
    assert A.dtype == torch.float32
    assert B.dtype == torch.float32
    assert A.is_contiguous()
    assert B.is_contiguous()

    # Allocate output tensor
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Define block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    # Define grid size
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
        1
    )

    # Launch the kernel
    matmul_kernel[grid](C, A, B, M, N, K,
                        C.stride(0), C.stride(1),
                        A.stride(0), A.stride(1),
                        B.stride(0), B.stride(1),
                        BLOCK_M, BLOCK_N, BLOCK_K)

    return C

# Example usage
if __name__ == "__main__":
    M, K, N = 1024, 1024, 1024
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    C = matmul(A, B)
    print(C)
