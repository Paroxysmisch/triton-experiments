import triton
import triton.language as tl

# Define constants for block sizes
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32

@triton.jit
def matmul_kernel(A, B, C, M, N, K, alpha, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # Define the block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the start of the block in the output matrix
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    # Initialize an accumulator for the block
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B into shared memory
        a_block = tl.load(A + (start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] * K + (k + tl.arange(0, BLOCK_SIZE_K)))
        b_block = tl.load(B + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * N + (start_n + tl.arange(0, BLOCK_SIZE_N)))

        # Compute the matrix multiplication for the block
        acc += tl.dot(a_block, b_block)

    # Apply the optional activation function (leaky_relu)
    acc = tl.where(acc > 0, acc, acc * alpha)

    # Write the result to the output matrix
    tl.store(C + (start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] * N + (start_n + tl.arange(0, BLOCK_SIZE_N)), acc)

def matmul(A, B, M, N, K, activation='none', alpha=0.01):
    # Ensure input matrices are compatible for multiplication
    assert A.shape == (M, K)
    assert B.shape == (K, N)

    # Allocate output matrix
    C = triton.empty((M, N), dtype=triton.float32)

    # Determine grid size based on block sizes
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch the kernel
    matmul_kernel[grid](
        A, B, C, M, N, K, alpha,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return C
