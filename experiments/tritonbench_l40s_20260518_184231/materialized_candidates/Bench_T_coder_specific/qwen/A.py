import triton
import triton.language as tl

@triton.jit
def solve_kernel(
    A_ptr, A_shape, A_strides,
    B_ptr, B_shape, B_strides,
    C_ptr, C_shape, C_strides,
    M: tl.int32, N: tl.int32, K: tl.int32,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    row = pid // grid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = pid % grid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Compute the inverse of A using LU decomposition
    pivot = tl.zeros((BLOCK_SIZE_M,), dtype=tl.int32)
    for k in range(BLOCK_SIZE_K):
        pivot[k] = k
        max_idx = tl.argmax(tl.abs(A_ptr[row, :, k]), axis=1) + k
        A_ptr[row, :, k], A_ptr[row, :, max_idx] = A_ptr[row, :, max_idx], A_ptr[row, :, k]
        for i in range(k+1, BLOCK_SIZE_M):
            A_ptr[row, i, k] /= A_ptr[row, k, k]
        for j in range(k+1, BLOCK_SIZE_N):
            for i in range(k+1, BLOCK_SIZE_M):
                A_ptr[row, i, j] -= A_ptr[row, i, k] * A_ptr[row, k, j]

    # Back substitution to find the inverse
    inv_A = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=A_ptr.dtype)
    for j in range(BLOCK_SIZE_N-1, -1, -1):
        inv_A[:, j] = A_ptr[:, j, :]
        for k in range(j+1, BLOCK_SIZE_N):
            inv_A[:, j] -= A_ptr[:, :, k] * inv_A[:, k]
        inv_A[:, j] /= A_ptr[:, j, j]

    # Perform the multiplication C = inv(A) @ B
    for m in range(BLOCK_SIZE_M):
        for n in range(BLOCK_SIZE_N):
            sum_ = 0.0
            for k in range(BLOCK_SIZE_K):
                sum_ += inv_A[m, k] * B_ptr[row[m], k, n]
            C_ptr[row[m], n] = sum_

# Triton wrapper function
def solve(A, B, *, left=False, out=None):
    A_shape = A.shape
    B_shape = B.shape
    assert len(A_shape) == 2 or len(A_shape) == 3, "Input A must be 2D or 3D"
    assert len(B_shape) == 2 or len(B_shape) == 3, "Input B must be 2D or 3D"
    assert A_shape[-2:] == B_shape[:-1], "Matrix dimensions do not match"

    if len(A_shape) == 2:
        M, N = A_shape
        K = B_shape[1]
        C_shape = (M, K)
    else:
        batch_size = A_shape[0]
        M, N = A_shape[1:]
        K = B_shape[2]
        C_shape = (batch_size, M, K)

    if out is None:
        out = torch.empty(C_shape, dtype=A.dtype, device=A.device)

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), 1)
    block = (BLOCK_SIZE_M, BLOCK_SIZE_N, 1)

    solve_kernel[grid, block](
        A.data_ptr(), A_shape, A.stride(),
        B.data_ptr(), B_shape, B.stride(),
        out.data_ptr(), C_shape, out.stride(),
        M, N, K,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )

    return out
