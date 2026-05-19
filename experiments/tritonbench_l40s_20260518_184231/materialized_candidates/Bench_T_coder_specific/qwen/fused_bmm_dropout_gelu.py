import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    x_ptr, y_ptr, z_ptr,
    x_batch_stride, x_row_stride, x_col_stride,
    y_batch_stride, y_row_stride, y_col_stride,
    z_batch_stride, z_row_stride, z_col_stride,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Each program will compute one block of C
    m = pid // grid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n = pid % grid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Block index within the MxN block computed by the program
    bm = tl.arange(0, BLOCK_SIZE_M)
    bn = tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        # Load x[bm, bk] and y[bk, bn]
        x = tl.load(x_ptr + x_batch_stride * pid + x_row_stride * bm[:, None] + x_col_stride * k[None, :])
        y = tl.load(y_ptr + y_batch_stride * pid + y_row_stride * k[None, :] + y_col_stride * bn)

        # Perform the dot product
        accumulator += tl.dot(x, y)

    # Write back the result
    tl.store(z_ptr + z_batch_stride * pid + z_row_stride * m[:, None] + z_col_stride * n, accumulator)
