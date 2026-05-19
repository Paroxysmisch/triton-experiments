import triton
import triton.language as tl

@triton.jit
def matrix_vector_dot_kernel(
    A_ptr, x_ptr, y_ptr,
    n, m, stride_A, stride_x, stride_y,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.program_id(1)

    acc = 0.0

    # Compute the matrix-vector product
    for k in range(m // BLOCK_SIZE_M):
        A_value = tl.load(A_ptr + row * stride_A + k * BLOCK_SIZE_M * stride_A, mask=row < n and k * BLOCK_SIZE_M < m, eviction_policy=tl.EVICT_FALSE)
        x_value = tl.load(x_ptr + k * BLOCK_SIZE_M * stride_x, mask=k * BLOCK_SIZE_M < m, eviction_policy=tl.EVICT_FALSE)
        acc += A_value * x_value

    # Update y in-place
    y_new = alpha * acc + beta * tl.load(y_ptr + row * stride_y, mask=row < n, eviction_policy=tl.EVICT_FALSE)
    tl.store(y_ptr + row * stride_y, y_new, mask=row < n)
