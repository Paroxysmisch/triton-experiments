import triton
import triton.language as tl
import kernel_utils

# Define the Triton kernel for matrix multiplication with extra elementwise operation
@triton.jit
def matmul_kernel(C, A, B, M, N, K,
                  stride_cm, stride_cn,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    # Triton kernel for matrix multiplication with extra elementwise operation
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

    # Perform the elementwise multiplication of the accumulator with itself
    c = kernel_utils.mul(accumulator, accumulator)

    # Write back the block of the output matrix C with masks
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

# Define the wrapper function for the Triton kernel
def matmul(A, B, C, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    # Validate input dimensions
    assert A.shape == (M, K), f"Matrix A has incorrect dimensions, expected ({M}, {K}), got {A.shape}"
    assert B.shape == (K, N), f"Matrix B has incorrect dimensions, expected ({K}, {N}), got {B.shape}"
    assert C.shape == (M, N), f"Matrix C has incorrect dimensions, expected ({M}, {N}), got {C.shape}"

    # Compute the grid size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch the kernel
    matmul_kernel[grid](C, A, B, M, N, K,
                        C.stride(0), C.stride(1),
                        A.stride(0), A.stride(1),
                        B.stride(0), B.stride(1),
                        BLOCK_M, BLOCK_N, BLOCK_K)

# Example usage
import torch

# Define the matrix dimensions
M, N, K = 1024, 1024, 1024

# Create the input matrices A and B
A = torch.randn((M, K), device='cuda')
B = torch.randn((K, N), device='cuda')

# Create the output matrix C
C = torch.empty((M, N), device='cuda')

# Define the block sizes
BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 16

# Call the matmul function
matmul(A, B, C, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)

# Print the result
print(C)
