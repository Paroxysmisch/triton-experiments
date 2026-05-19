import triton
import triton.language as tl

# Define the kernel
@triton.jit
def matmul_kernel(C, A, B, M, N, K,
                  stride_cm, stride_cn,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr,
                  ACTIVATION: tl.constexpr):
    # Triton kernel for matrix multiplication with optional activation
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the offsets for the current block
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
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
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Apply optional activation (leaky ReLU)
    if ACTIVATION:
        accumulator = tl.where(accumulator > 0, accumulator, 0.01 * accumulator)

    # Perform element-wise multiplication C = C * C
    c = accumulator * accumulator

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

# Define the high-level interface
def matmul(A, B, C, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, activation=False):
    # Ensure input validity
    assert A.shape == (M, K), f"Matrix A must have shape (M, K), but got {A.shape}"
    assert B.shape == (K, N), f"Matrix B must have shape (K, N), but got {B.shape}"
    assert C.shape == (M, N), f"Matrix C must have shape (M, N), but got {C.shape}"

    # Prepare execution parameters
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), 1)
    ACTIVATION = 1 if activation else 0

    # Invoke the kernel
    matmul_kernel[grid](C, A, B, M, N, K,
                        C.stride(0), C.stride(1),
                        A.stride(0), A.stride(1),
                        B.stride(0), B.stride(1),
                        BLOCK_M, BLOCK_N, BLOCK_K, ACTIVATION)

# Example usage
import torch

# Define matrix dimensions
M, N, K = 1024, 1024, 1024
BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 16

# Create input tensors
A = torch.randn((M, K), device='cuda')
B = torch.randn((K, N), device='cuda')
C = torch.empty((M, N), device='cuda')

# Perform matrix multiplication with optional activation
matmul(A, B, C, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, activation=True)

# Print the result
print(C)
