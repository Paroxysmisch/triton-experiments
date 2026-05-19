import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(
    input_ptr, output_ptr, M, N, TILE_N: tl.constexpr
):
    # Calculate the tile index
    tile_m = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 tile
    tile_start_ptr = input_ptr + tile_m * TILE_N * M
    # Load the tile into SRAM
    tile = tl.load(tile_start_ptr, (M, TILE_N))

    # Maximum value reduction
    max_val = tl.max(tile, axis=1)
    # Maximum value subtraction
    tile_red = tile - max_val[:, None]
    # Softmax numerator computation
    numerator = tl.exp(tile_red)
    # Softmax denominator computation
    denominator = tl.sum(numerator, axis=1)
    # Softmax computation
    softmax_val = numerator / denominator[:, None]
    # Store output
    tl.store(output_ptr + tile_m * TILE_N * M, softmax_val, (M, TILE_N))

def softmax(x):
    M, N = x.shape
    # Calculate the size of the tile
    TILE_N = min(M, triton.next_power_of_2(M))
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel.
    softmax_kernel_online_v2[(M + TILE_N - 1) // TILE_N,](y, x, M, N, TILE_N=TILE_N)
    return y
