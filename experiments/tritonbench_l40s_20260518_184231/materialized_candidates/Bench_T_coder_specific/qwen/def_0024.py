import triton
import triton.language as tl

@triton.jit
def qr_decomposition_kernel(
    A_ptr,
    Q_ptr,
    R_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Load A into shared memory
    A_shared = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if row < M and col < N:
        A_shared[row, col] = tl.load(A_ptr + row * N + col)

    tl.sync_mem()

    # Perform QR decomposition
    for j in range(N):
        if col == j:
            r_norm = tl.math.sqrt(tl.sum(A_shared[:, j] * A_shared[:, j]))
            Q_shared = A_shared / r_norm
            R_shared = r_norm
        else:
            r_norm = tl.math.sqrt(tl.sum(Q_shared[:j+1, j] * Q_shared[:j+1, j]))
            c = Q_shared[:j+1, j] / r_norm
            s = -c[0]
            Q_shared[:j+1, j] = tl.where(tl.arange(j+1) < j, Q_shared[:j+1, j], 0.0)
            Q_shared[:j+1, j:] = Q_shared[:j+1, j:] @ tl.transpose(tl.stack([c, s]))

        if row < M and col < N:
            A_shared[row, col] -= Q_shared[:j+1, col] @ Q_shared[:j+1, j:]
        tl.sync_mem()

    # Store Q and R back to global memory
    if row < M and col < N:
        tl.store(Q_ptr + row * N + col, Q_shared[row, col])
        tl.store(R_ptr + row * N + col, R_shared)

# Function to perform QR decomposition
def qr_decomposition(A):
    M, N = A.shape[-2:]
    grid = ((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M, (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    block = (BLOCK_SIZE_M, BLOCK_SIZE_N)
    Q = tl.zeros_like(A)
    R = tl.zeros_like(A)
    qr_decomposition_kernel[grid, block](A.data_ptr(), Q.data_ptr(), R.data_ptr(), M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    return Q, R
