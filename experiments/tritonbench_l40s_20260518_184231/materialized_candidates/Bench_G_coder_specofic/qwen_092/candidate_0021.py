triton
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, O_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr, DIVISIBLE_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    batch: tl.constexpr,
    grid_idx: tl.constexpr
):
    # Determine the tile indices
    tile_m = grid_idx // (N // TILE_N)
    tile_n = (grid_idx % (N // TILE_N)) // (M // TILE_M)
    tile_k = (grid_idx % (N // TILE_N)) % (M // TILE_M)

    # Determine the offsets within the tile
    m = tile_m * TILE_M + tl.program_id(1)
    n = tile_n * TILE_N + tl.program_id(0)
    k = tile_k * TILE_K

    # Initialize the output tile
    output_tile = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Load tiles of A and B
    A_tile = tl.load(A_ptr + m * M * K + k * K, mask=m < M and k < K, other=0.0)
    B_tile = tl.load(B_ptr + n * K * N + k * N, mask=k < K and n < N, other=0.0)

    # Perform the matrix multiplication
    for i in range(TILE_K):
        output_tile += A_tile[:, i:i+1] * B_tile[i:i+1, :]

    # Store the result in the output tensor
    tl.store(O_ptr + m * M * N + n * N, output_tile, mask=m < M and n < N)
