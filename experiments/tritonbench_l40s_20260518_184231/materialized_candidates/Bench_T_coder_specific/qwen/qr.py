import triton
import triton.language as tl

@triton.jit
def qr_kernel(
    A_ptr,
    Q_ptr,
    R_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    row_base = row_idx * BLOCK_SIZE_M
    col_base = col_idx * BLOCK_SIZE_N

    for k in range(K):
        # Compute the Householder vector h_k
        sum_squares = 0.0
        for i in range(row_base + k, min(row_base + M, row_base + k + BLOCK_SIZE_M)):
            sum_squares += A_ptr[i * N + col_base + k] ** 2

        h_k = tl.sqrt(sum_squares)
        alpha = -tl.sign(A_ptr[row_base + k * N + col_base + k]) * h_k
        A_ptr[row_base + k * N + col_base + k] -= alpha

        # Apply Householder transformation to A
        for i in range(row_base + k + 1, min(row_base + M, row_base + k + BLOCK_SIZE_M)):
            v_i = A_ptr[i * N + col_base + k]
            A_ptr[i * N + col_base + k] = 0.0
            for j in range(col_base + k + 1, min(col_base + N, col_base + k + BLOCK_SIZE_N)):
                A_ptr[i * N + j] -= (v_i * A_ptr[row_base + k * N + j]) / alpha
                A_ptr[row_base + k * N + j] -= (v_i * A_ptr[i * N + j]) / alpha

        # Update Q
        if Q_ptr is not None:
            for i in range(row_base + k, min(row_base + M, row_base + k + BLOCK_SIZE_M)):
                q_i = Q_ptr[i * N + col_base + k]
                Q_ptr[i * N + col_base + k] = 0.0
                for j in range(col_base + k + 1, min(col_base + N, col_base + k + BLOCK_SIZE_N)):
                    Q_ptr[i * N + j] -= (q_i * A_ptr[row_base + k * N + j]) / alpha
                    Q_ptr[i * N + j] -= (q_i * A_ptr[i * N + j]) / alpha

        # Update R
        for j in range(col_base + k + 1, min(col_base + N, col_base + k + BLOCK_SIZE_N)):
            r_j = R_ptr[col_base + k * N + j]
            R_ptr[col_base + k * N + j] = 0.0
            for i in range(row_base + k + 1, min(row_base + M, row_base + k + BLOCK_SIZE_M)):
                R_ptr[i * N + j] -= (A_ptr[i * N + j] * A_ptr[row_base + k * N + j]) / alpha
                R_ptr[i * N + j] -= (A_ptr[i * N + j] * A_ptr[i * N + j]) / alpha

# Define the block sizes
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32

# Example usage
M = 4
N = 5
K = min(M, N)
A = tl.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=tl.float32)
Q = tl.zeros((M, N), dtype=tl.float32)
R = tl.zeros((N, N), dtype=tl.float32)

qr_kernel[(M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)](A.data_ptr(), Q.data_ptr(), R.data_ptr(), M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N)
