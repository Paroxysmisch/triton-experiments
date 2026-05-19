import torch
import triton
import triton.language as tl
from ..utils.shape_utils import prev_multiple_of


@triton.jit
def softmax_kernel_online_v2(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
):
    # Compute the offset for the current program ID
    program_id = tl.program_id(0)
    input_row_off = program_id * TILE_N
    output_row_off = program_id * TILE_N
    row_off = input_row_off

    # Compute the maximum value for the stable softmax
    max_value = tl.full([TILE_N], float("-inf"), dtype=tl.float32)
    row_mask = (row_off + tl.arange(0, TILE_N)) < M

    for col_off in range(0, N, TILE_N):
        col_mask = (col_off + tl.arange(0, TILE_N)) < N
        mask = row_mask and col_mask

        input_ptr_masked = input_ptr + (row_off + tl.arange(0, TILE_N))[:, None] * N + col_off + tl.arange(0, TILE_N)[None, :]
        input_row = tl.load(input_ptr_masked, mask=mask, other=0.0)
        curr_max = tl.max(input_row, axis=1)
        max_value = tl.maximum(curr_max, max_value)

    # Compute the sum of exponentials for the stable softmax
    sum_exp = tl.full([TILE_N], 0.0, dtype=tl.float32)
    for col_off in range(0, N, TILE_N):
        col_mask = (col_off + tl.arange(0, TILE_N)) < N
        mask = row_mask and col_mask

        input_ptr_masked = input_ptr + (row_off + tl.arange(0, TILE_N))[:, None] * N + col_off + tl.arange(0, TILE_N)[None, :]
        input_row = tl.load(input_ptr_masked, mask=mask, other=0.0)
        curr_max = tl.max(input_row, axis=1)
        input_row = input_row - curr_max[:, None]
        exp_row = tl.exp(input_row)
        sum_exp_row = tl.sum(exp_row, axis=1)
        sum_exp = sum_exp + sum_exp_row

    # Compute the stable softmax
    for col_off in range(0, N, TILE_N):
        col_mask = (col_off + tl.arange(0, TILE_N)) < N
        mask = row_mask and col_mask

        input_ptr_masked = input_ptr + (row_off + tl.arange(0, TILE_N))[:, None] * N + col_off + tl.arange(0, TILE_N)[None, :]
        output_ptr_masked = output_ptr + (output_row_off + tl.arange(0, TILE_N))[:, None] * N + col_off + tl.arange(0, TILE_N)[None, :]
        input_row = tl.load(input_ptr_masked, mask=mask, other=0.0)
        curr_max = tl.max(input_row, axis=1)
        input_row = input_row - curr_max[:, None]
        exp_row = tl.exp(input_row)
        norm_row = exp_row / sum_exp
        tl.store(output_ptr_masked, norm_row, mask=mask)

    return


@torch.no_grad()
def softmax(input: torch.Tensor) -> torch.Tensor:
    if input.dtype == torch.float16:
        input = input.to(torch.bfloat16)

    M = input.shape[0]
    N = input.shape[1]

    TILE_N = 16
    grid = (triton.cdiv(M, TILE_N), 1, 1)

    softmax_out = torch.empty_like(input)

    softmax_kernel_online_v2[grid](
        softmax_out,
        input,
        M,
        N,
        TILE_N,
    )

    return softmax_out
