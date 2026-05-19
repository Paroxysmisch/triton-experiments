import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    x_ptr, y_ptr, z_ptr,
    M, N, K,
    stride_x_M, stride_x_K,
    stride_y_N, stride_y_K,
    stride_z_M, stride_z_N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_blocks_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_blocks_N = tl.cdiv(N, BLOCK_SIZE_N)

    # Compute the coordinates within the block
    block_m = pid // num_blocks_N * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    block_n = pid % num_blocks_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize accumulation buffer
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        # Load x and y chunks
        x_chunk = tl.load(x_ptr + block_m[:, None] * stride_x_M + (k + tl.arange(0, BLOCK_SIZE_K)) * stride_x_K, mask=(block_m[:, None] < M) & (k + tl.arange(0, BLOCK_SIZE_K) < K))
        y_chunk = tl.load(y_ptr + (k + tl.arange(0, BLOCK_SIZE_K)) * stride_y_K + block_n[None, :], mask=(k + tl.arange(0, BLOCK_SIZE_K) < K) & (block_n[None, :] < N))

        # Perform the matrix multiplication
        acc += tl.dot(x_chunk, y_chunk)

    # Write back the result
    tl.store(z_ptr + block_m[:, None] * stride_z_M + block_n[None, :], acc, mask=(block_m[:, None] < M) & (block_n[None, :] < N))
