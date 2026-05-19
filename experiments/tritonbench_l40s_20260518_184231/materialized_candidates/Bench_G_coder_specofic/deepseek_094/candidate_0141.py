import triton
import triton.language as tl

@triton.jit
def matmul_kernel_persistent(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a, stride_b, stride_c,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    num_warps, num_stages
):
    # Define the block index and warp index
    block_idx = tl.program_id(axis=0)
    warp_idx = block_idx % num_warps

    # Define the tile indices
    tile_i = warp_idx // num_stages
    tile_j = warp_idx % num_stages

    # Define the offsets for the input and output matrices
    offset_a = block_idx * BLOCK_SIZE_M * stride_a
    offset_b = tile_j * BLOCK_SIZE_N * stride_b
    offset_c = tile_i * BLOCK_SIZE_M * stride_c

    # Define the block index and warp index
    block_idx = tl.program_id(axis=0)
    warp_idx = block_idx % num_warps

    # Define the tile indices
    tile_i = warp_idx // num_stages
    tile_j = warp_idx % num_stages

    # Define the offsets for the input and output matrices
    offset_a = block_idx * BLOCK_SIZE_M * stride_a
    offset_b = tile_j * BLOCK_SIZE_N * stride_b
    offset_c = tile_i * BLOCK_SIZE_M * stride_c

    # Load the input matrices
    a = tl.load(a_ptr + offset_a, mask=(tile_i < M and tile_j < K))
    b = tl.load(b_ptr + offset_b, mask=(tile_i < K and tile_j < N))

    # Initialize the accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Perform the matrix multiplication
    for k in range(K):
        accumulator += a[tile_i, k] * b[k, tile_j]

    # Store the output matrix
    c = tl.load(c_ptr + offset_c)
    c += accumulator
    tl.store(c_ptr + offset_c, c)


def matmul_persistent(a, b, c, dtype=tl.float32):
    # Validate the input tensors
    assert a.ndim == 2 and b.ndim == 2 and c.ndim == 2
    assert a.shape[1] == b.shape[0] and a.shape[0] == c.shape[0] and b.shape[1] == c.shape[1]
    assert a.dtype == dtype and b.dtype == dtype and c.dtype == dtype

    # Compute the grid size
    M, K = a.shape
    N, _ = b.shape
    _, N = c.shape
    grid = lambda M, N: ((M - 1) // BLOCK_SIZE_M + 1, (N - 1) // BLOCK_SIZE_N + 1)
    grid_a, grid_b = grid(M, K), grid(K, N)

    # Compute the strides
    stride_a = a.strides[0]
    stride_b = b.strides[0]
    stride_c = c.strides[0]

    # Configure the kernel parameters
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 16, 16, 16
    num_warps = 4
    num_stages = 2

    # Launch the kernel
    matmul_kernel_persistent[grid_a, BLOCK_SIZE_M](
        a.gpu_data_ptr, b.gpu_data_ptr, c.gpu_data_ptr,
        M, N, K,
        stride_a, stride_b, stride_c,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        num_warps, num_stages
    )

    return c
