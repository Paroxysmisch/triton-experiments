import torch
import triton
import triton.language as tl

@triton.jit
def prev_multiple_of(x, base: tl.constexpr):
    return (x // base) * base

@triton.jit
def softmax_kernel_online_v2(
    output_ptr, input_ptr,
    M, N,
    TILE_N: tl.constexpr,
    input_row_stride, output_row_stride
):
    row_id = tl.program_id(0)
    # Pointers for this row
    row_in_ptr = input_ptr + row_id * input_row_stride
    row_out_ptr = output_ptr + row_id * output_row_stride

    # Phase 1: Compute row-wise max
    row_max = -float('inf')
    tile_aligned_end = prev_multiple_of(N, TILE_N)
    offset = tl.arange(0, TILE_N)

    # Tiled pass for max
    ptrs = row_in_ptr + offset
    col_idx = offset
    mask = col_idx < N
    tile_max = tl.load(ptrs, mask=mask, other=-float('inf'))
    row_max = tl.maximum(row_max, tl.max(tile_max, axis=0))
    tile_start = TILE_N
    while tile_start < tile_aligned_end:
        ptrs = row_in_ptr + tile_start + offset
        col_idx = tile_start + offset
        mask = col_idx < N
        tile_vals = tl.load(ptrs, mask=mask, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(tile_vals, axis=0))
        tile_start += TILE_N

    # Handle leftover columns for max
    leftover_start = tile_aligned_end
    if leftover_start < N:
        ptrs = row_in_ptr + leftover_start + offset
        col_idx = leftover_start + offset
        mask = col_idx < N
        tile_vals = tl.load(ptrs, mask=mask, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(tile_vals, axis=0))

    # Phase 2: Compute sum of exponentials, then write final softmax
    row_sum = 0.0

    # Tiled pass for sum
    ptrs = row_in_ptr + offset
    col_idx = offset
    mask = col_idx < N
    tile_vals = tl.load(ptrs, mask=mask, other=0.0)
    tile_exp = tl.exp(tile_vals - row_max)
    row_sum += tl.sum(tile_exp, axis=0)
    tile_start = TILE_N
    while tile_start < tile_aligned_end:
        ptrs = row_in_ptr + tile_start + offset
        col_idx = tile_start + offset
        mask = col_idx < N
        tile_vals = tl.load(ptrs, mask=mask, other=0.0)
        tile_exp = tl.exp(tile_vals - row_max)
        row_sum += tl.sum(tile_exp, axis=0)
        tile_start += TILE_N

    # Handle leftover columns for sum
    leftover_start = tile_aligned_end
    if leftover_start < N:
        ptrs = row_in_ptr + leftover_start + offset
        col_idx = leftover_start + offset
        mask = col_idx < N
        tile_vals = tl.load(ptrs, mask=mask, other=0.0)
        tile_exp = tl.exp(tile_vals - row_max)
        row_sum += tl.sum(tile_exp, axis=0)

    # Final pass: write results
    # Tiled pass
    ptrs_in = row_in_ptr + offset
    ptrs_out = row_out_ptr + offset
    col_idx = offset
    mask = col_idx < N
    tile_vals = tl.load(ptrs_in, mask=mask, other=0.0)
    tile_exp = tl.exp(tile_vals - row_max)
    tile_softmax = tile_exp / row_sum
    tl.store(ptrs_out, tile_softmax, mask=mask)
    tile_start = TILE_N
    while tile_start < tile_aligned_end:
        ptrs_in = row_in_ptr + tile_start + offset
        ptrs_out = row_out_ptr + tile
