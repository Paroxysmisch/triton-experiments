import torch
import triton
import triton.language as tl

@triton.jit
def prev_multiple_of(a, b: tl.constexpr):
    return (a // b) * b

@triton.jit
def softmax_kernel_online_v2(output_ptr, input_ptr, M, N, TILE_N: tl.constexpr):
    # Get the row index for the current program instance
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    # Compute the starting pointer for the current row
    row_start_ptr = input_ptr + row_idx * N

    # Compute the column offsets for the current tile
    col_offsets = tl.arange(0, TILE_N)
    input_ptrs = row_start_ptr + col_offsets

    # Load the tile into SRAM, using a mask since TILE_N may be > than N
    tile = tl.load(input_ptrs, mask=col_offsets < N, other=-float('inf'))

    # Subtract the maximum value for numerical stability
    max_val = tl.max(tile, axis=0)
    tile_minus_max = tile - max_val

    # Compute the exponentials
    exp_tile = tl.exp(tile_minus_max)

    # Compute the sum of the exponentials
    sum_exp = tl.sum(exp_tile, axis=0)

    # Compute the softmax values
    softmax_tile = exp_tile / sum_exp

    # Write back the softmax values to global memory
    output_row_start_ptr = output_ptr + row_idx * N
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_tile, mask=col_offsets < N)

def softmax(x):
    M, N = x.shape
    # Determine the tile size, which is a power of 2
    TILE_N = triton.next_power_of_2(N)
    # Allocate output tensor
    y = torch.empty_like(x)
    # Determine the number of warps
    num_warps = 4
    if TILE_N >= 2048:
        num_warps = 8
    if TILE_N >= 4096:
        num_warps = 16
    # Enqueue the kernel
    softmax_kernel_online_v2[(M,)](
        y,
        x,
        M,
        N,
        TILE_N,
        num_warps=num_warps
    )
    return y
