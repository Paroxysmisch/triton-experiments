import triton
import triton.language as tl

# Define the block size
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 16

# Define the kernel
@triton.jit
def matmul_kernel(
    A, B, C,
    M: tl.int32, K: tl.int32, N: tl.int32,
    BLOCK_SIZE_M: tl.int32, BLOCK_SIZE_N: tl.int32, BLOCK_SIZE_K: tl.int32,
    ACTIVATION: tl.bool
):
    # Define program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Define indices
    i = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    j = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Load blocks of A and B
    a = tl.load(A + i[:, None] * K + k, mask=i[:, None] < M, mask_value=0.0)
    b = tl.load(B + k[:, None] * N + j, mask=k[:, None] < K, mask_value=0.0)

    # Perform matrix multiplication
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A + i[:, None] * K + k, mask=i[:, None] < M, mask_value=0.0)
        b = tl.load(B + k[:, None] * N + j, mask=k[:, None] < K, mask_value=0.0)
        accumulator += a[:, None] * b

    # Apply leaky ReLU activation if required
    if ACTIVATION:
        accumulator = tl.maximum(accumulator, 0.01 * accumulator)

    # Store the result in C
    tl.store(C + i[:, None] * N + j, accumulator, mask=i[:, None] < M & j < N)

# Define the wrapper function
def matmul(A, B, C, M, K, N, ACTIVATION=False):
    # Check input dimensions
    assert A.shape == (M, K), "Invalid shape for A"
    assert B.shape == (K, N), "Invalid shape for B"
    assert C.shape == (M, N), "Invalid shape for C"

    # Set Triton kernel parameters
    grid = (tl.cdiv(M, BLOCK_SIZE_M), tl.cdiv(N, BLOCK_SIZE_N))
    block = (BLOCK_SIZE_M, BLOCK_SIZE_N, 1)

    # Launch the kernel
    matmul_kernel[grid, block](A, B, C, M, K, N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, ACTIVATION)

# Example usage
if __name__ == "__main__":
    import numpy as np

    M, K, N = 128, 256, 512
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    matmul(A, B, C, M, K, N, ACTIVATION=True)

    print("C shape:", C.shape)
    print("C[:5, :5]:", C[:5, :5])
