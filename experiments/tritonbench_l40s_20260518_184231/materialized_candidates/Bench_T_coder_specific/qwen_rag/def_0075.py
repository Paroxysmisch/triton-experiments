import triton
import triton.language as tl

@triton.jit
def cholesky_kernel(A_ptr, L_ptr, n: tl.int32, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Cholesky decomposition
    for j in range(row, n, BLOCK_SIZE):
        sum_k = 0.0
        for k in range(j):
            sum_k += L_ptr[j * n + k] * L_ptr[j * n + k]
        L_ptr[j * n + j] = tl.sqrt(A_ptr[j * n + j] - sum_k)

        for i in range(j + 1, n):
            sum_k = 0.0
            for k in range(j):
                sum_k += L_ptr[i * n + k] * L_ptr[j * n + k]
            L_ptr[i * n + j] = (A_ptr[i * n + j] - sum_k) / L_ptr[j * n + j]

# Wrapper function to invoke the Cholesky kernel
def cholesky(A: tl.tensor) -> tl.tensor:
    n = A.shape[0]
    L = tl.zeros((n, n), dtype=A.dtype)
    block_size = min(128, n)
    grid_size = (tl.cdiv(n, block_size), tl.cdiv(n, block_size))

    cholesky_kernel[(grid_size, grid_size)](A.data, L.data, n, block_size)
    return L
