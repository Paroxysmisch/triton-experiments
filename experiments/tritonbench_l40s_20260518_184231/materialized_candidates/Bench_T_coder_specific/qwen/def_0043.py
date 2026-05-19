import triton
import triton.language as tl

@triton.jit
def symmetric_matrix_vector_kernel(
    A_ptr, x_ptr, y_ptr, n, alpha, beta,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    row_end = min(row_start + BLOCK_SIZE, n)

    # Load data into shared memory
    x_shared = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    if row_start < n:
        x_shared[pid % BLOCK_SIZE] = tl.load(x_ptr + row_start)
    
    A_shared = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        if row_start + i < n and pid < n:
            A_shared[i, :] = tl.load(A_ptr + (row_start + i) * n)

    # Perform matrix-vector multiplication
    y_val = 0.0
    for j in range(BLOCK_SIZE):
        y_val += A_shared[j, :] @ x_shared[j]

    # Accumulate result
    tl.store(y_ptr + row_start, y_val + beta * tl.load(y_ptr + row_start))
