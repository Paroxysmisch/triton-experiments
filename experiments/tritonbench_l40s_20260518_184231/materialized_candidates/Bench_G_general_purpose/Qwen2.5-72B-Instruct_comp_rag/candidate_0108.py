import torch
import triton
import triton.language as tl

# Helper function to compute the largest multiple of `tile_size` less than or equal to `n`
@triton.jit
def prev_multiple_of(n, tile_size):
    return (n // tile_size) * tile_size

@triton.jit
def softmax_kernel_online_v2(output_ptr, input_ptr, M, N, TILE_N: tl.constexpr):
    # Get the program ID for the current block
    pid = tl.program_id(0)
    
    # Compute the number of tiles in the row direction
    num_tiles = (N + TILE_N - 1) // TILE_N
    
    # Compute the starting row and column for the current block
    row_start = (pid // num_tiles) * TILE_N
    col_start = (pid % num_tiles) * TILE_N
    
    # Compute the end of the tile, ensuring it does not exceed the matrix dimensions
    row_end = min(row_start + TILE_N, M)
    col_end = min(col_start + TILE_N, N)
    
    # Compute the number of elements in the current tile
    tile_size = (row_end - row_start) * (col_end - col_start)
    
    # Allocate shared memory for the tile
    tile = tl.zeros((TILE_N, TILE_N), dtype=tl.float32)
    
    # Load the tile from global memory to shared memory
    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            tile[i - row_start, j - col_start] = tl.load(input_ptr + i * N + j)
    
    # Compute the maximum value in the tile for numerical stability
    max_val = tl.max(tile, axis=None)
    
    # Subtract the maximum value from each element in the tile
    tile -= max_val
    
    # Compute the exponentials
    exp_tile = tl.exp(tile)
    
    # Compute the sum of the exponentials
    sum_exp = tl.sum(exp_tile, axis=None)
    
    # Compute the softmax values
    softmax_tile = exp_tile / sum_exp
    
    # Write the softmax values back to global memory
    for i in range(row_start, row_end):
        for j in range(col_start, col_end):
            tl.store(output_ptr + i * N + j, softmax_tile[i - row_start, j - col_start])

def softmax(x, TILE_N=128):
    M, N = x.shape
    # Allocate output
    y = torch.empty_like(x)
    
    # Determine the number of blocks needed
    num_tiles = (N + TILE_N - 1) // TILE_N
    num_blocks = (M + TILE_N - 1) // TILE_N * num_tiles
    
    # Enqueue the kernel
    softmax_kernel_online_v2[(num_blocks,)](
        y,
        x,
        M,
        N,
        TILE_N=TILE_N,
        num_warps=4  # Adjust num_warps based on the input size if needed
    )
    
    return y
