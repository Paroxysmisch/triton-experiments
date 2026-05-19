import triton
import triton.language as tl

@triton.jit
def softmax_kernel_online_v2(
    input_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    M,  # Number of rows in the input tensor
    N,  # Number of columns in the input tensor
    TILE_N: tl.constexpr,  # Tile size for the columns
):
    # Compute the block ID in the grid
    pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(N, TILE_N)
    tile_id = pid % num_tiles
    row_id = pid // num_tiles

    # Compute the start and end indices for the current tile
    tile_start = tile_id * TILE_N
    tile_end = min(tile_start + TILE_N, N)

    # Compute the row offset
    row_offset = row_id * N

    # Load the input data for the current tile
    input_tile = tl.load(input_ptr + row_offset + tile_start, mask=tile_start + tl.arange(0, TILE_N) < N, other=-float('inf'))

    # Compute the maximum value in the current tile
    max_val = tl.max(input_tile, axis=0)

    # Compute the exponentials
    exp_val = tl.exp(input_tile - max_val)

    # Compute the sum of exponentials
    sum_exp = tl.sum(exp_val, axis=0)

    # Normalize the exponentials to get the softmax values
    output_tile = exp_val / sum_exp

    # Store the output data for the current tile
    tl.store(output_ptr + row_offset + tile_start, output_tile, mask=tile_start + tl.arange(0, TILE_N) < N)

import torch
import triton
import triton.language as tl

def softmax(input_tensor: torch.Tensor, TILE_N: int = 128) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    input_tensor = input_tensor.cuda()
    M, N = input_tensor.shape

    # Allocate memory for the output tensor
    output_tensor = torch.empty_like(input_tensor)

    # Define the grid and block dimensions
    grid = (triton.cdiv(M * N, TILE_N),)
    block = (TILE_N,)

    # Launch the kernel
    softmax_kernel_online_v2[grid, block](
        input_tensor,  # Pointer to the input tensor
        output_tensor,  # Pointer to the output tensor
        M,  # Number of rows in the input tensor
        N,  # Number of columns in the input tensor
        TILE_N  # Tile size for the columns
    )

    return output_tensor
