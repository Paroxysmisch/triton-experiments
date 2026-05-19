triton
@triton.jit
def _int8_matmul_rowwise_dequantize(
    a_ptr: ptr32, a_stride0: ptr32, a_stride1: ptr32,
    b_ptr: ptr32, b_stride0: ptr32, b_stride1: ptr32,
    c_ptr: ptr32, c_stride0: ptr32, c_stride1: ptr32,
    state_x_ptr: ptr32, state_x_scale_ptr: ptr32,
    state_w_ptr: ptr32, state_w_scale_ptr: ptr32,
    bias_ptr: ptr32, bias_stride0: ptr32,
    M: int32, N: int32, K: int32,
    BLOCK_M: int32, BLOCK_N: int32, BLOCK_K: int32,
    SPLIT_K: int32,
    grid: tl.program_id(2)
):
    # Define block indices
    block_row = tl.program_id(0)
    block_col = tl.program_id(1)
    block_k = grid

    # Define offsets
    row_start = block_row * BLOCK_M
    col_start = block_col * BLOCK_N
    k_start = block_k * BLOCK_K

    # Allocate shared memory
    a_block = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.int32)
    b_block = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.int32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Load A and B blocks into shared memory
    for k in range(0, BLOCK_K, BLOCK_K // SPLIT_K):
        a_block_row = row_start + tl.arange(0, BLOCK_M)
        a_block_col = k_start + tl.arange(0, BLOCK_K // SPLIT_K)
        a_block[a_block_row, a_block_col] = tl.load(a_ptr + a_block_row * a_stride0 + a_block_col * a_stride1, mask=(a_block_row < M) & (a_block_col < K // SPLIT_K), other=0)

        b_block_col = col_start + tl.arange(0, BLOCK_N)
        b_block_row = k_start + tl.arange(0, BLOCK_K // SPLIT_K)
        b_block[b_block_row, b_block_col] = tl.load(b_ptr + b_block_row * b_stride0 + b_block_col * b_stride1, mask=(b_block_row < K // SPLIT_K) & (b_block_col < N), other=0)

        # Perform matrix multiplication and accumulation
        for k in range(BLOCK_K // SPLIT_K):
            a_block_row = row_start + tl.arange(0, BLOCK_M)
            a_block_col = k_start + k
            b_block_col = col_start + tl.arange(0, BLOCK_N)
            a_block_row = row_start + tl.arange(0, BLOCK_M)
            a_block_col = k_start + k
            acc[a_block_row, b_block_col] += tl.dot(a_block[a_block_row, a_block_col], b_block[a_block_col, b_block_col], dtype=tl.int32)

    # Dequantize and scale
    for m in range(BLOCK_M):
        for n in range(BLOCK_N):
            a_val = a_block[m, 0]
            b_val = b_block[0, n]
            x_scale = tl.load(state_x_scale_ptr + a_val * 4)
            w_scale = tl.load(state_w_scale_ptr + b_val * 4)
            acc[m, n] = tl.cast(acc[m, n], tl.float32) * x_scale * w_scale

    # Add bias if present
    if bias_ptr is not None:
        for m in range(BLOCK_M):
            for n in range(BLOCK_N):
                acc[m, n] += tl.load(bias_ptr + m * bias_stride0 + n * 4, mask=(m < M) & (n < N), other=0)

    # Write to C matrix
    for m in range(BLOCK_M):
        for n in range(BLOCK_N):
            c_ptr[row_start + m * c_stride0 + col_start + n * c_stride1] = tl.atomic_add(c_ptr[row_start + m * c_stride0 + col_start + n * c_stride1], tl.cast(acc[m, n], tl.int32))
