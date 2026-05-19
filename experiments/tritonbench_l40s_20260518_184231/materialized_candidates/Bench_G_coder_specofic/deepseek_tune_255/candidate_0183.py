import math
import torch
import triton
import triton.language as tl

@triton.heuristics(
    {
        "TILE_K": lambda args: triton.next_power_of_2(args["K"]),
        "TILE_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.jit
def softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if ONE_TILE_PER_CTA:
        tile_n_offset = pid_m * TILE_N
        tile_k_offset = pid_k * TILE_K
        input_ptr += tile_n_offset * K + tile_k_offset
        output_ptr += tile_n_offset * K + tile_k_offset
    else:
        tile_k_offset = pid_k * TILE_K
        input_ptr += tile_k_offset
        output_ptr += tile_k_offset

    offs_m = pid_m * TILE_N + tl.arange(0, TILE_N)
    offs_n = tl.arange(0, TILE_K)
    idx = offs_m[:, None] * K + offs_n[None, :]

    mask = offs_n[None, :] < K - tile_k_offset

    logits = tl.load(input_ptr + idx, mask=mask, other=-float("inf")).to(tl.float32)
    logits -= tl.max(logits, axis=1)[:, None]
    numerator = tl.exp(logits)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = numerator / denominator

    tl.store(output_ptr + idx, softmax_output, mask=mask)

@triton.heuristics(
    {
        "TILE_K": lambda args: triton.next_power_of_2(args["K"]),
        "TILE_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Inner softmax kernel.
    Special case of softmax where K is in the inner dimension.
    TILE_N is the number of inner elements processed by a single CTA.
    """
    # Infer the number of tiles based on the number of elements.
    # There's no point in doing a tile if the number of elements is less than the tile size.
    # This means that we only have 1 tile.
    GROUP_SIZE_N = TILE_N
    NUM_TILES = triton.cdiv(N, TILE_N)

    pid = tl.program_id(0)
    # group size of 1 is a nice property that it doesn't need to worry about
    # data dependency issues because as soon as it reads the data it gets
    # stalled and hence the kernel will issue more work
    group_id = pid % GROUP_SIZE_N
    row_start = group_id * GROUP_SIZE_M
    # Each row in the grid starts at the same offset.
    input_ptr += row_start * K
    output_ptr += row_start * K

    col_offsets = (pid // GROUP_SIZE_N) * TILE_K
    offs_m = row_start + tl.arange(0, GROUP_SIZE_M)
    offs_n = col_offsets + tl.arange(0, TILE_K)
    idx = offs_m[:, None] * K + offs_n[None, :]

    mask = offs_n[None, :] < K - col_offsets

    logits = tl.load(input_ptr + idx, mask=mask, other=-float("inf")).to(tl.float32)
    logits -= tl.max(logits, axis=1)[:, None]
    numerator = tl.exp(logits)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = numerator / denominator

    tl.store(output_ptr + idx, softmax_output, mask=mask)

@triton.heuristics(
    {
        "TILE_K": lambda args: triton.next_power_of_2(args["K"]),
        "TILE_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.jit
def softmax_backward_kernel_non_inner(
    out_grad_ptr,
    in_grad_ptr,
    output_ptr,
    M,
    N,
    K,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if ONE_TILE_PER_CTA:
        tile_n_offset = pid_m * TILE_N
        tile_k_offset = pid_k * TILE_K
        out_grad_ptr += tile_n_offset * K + tile_k_offset
        in_grad_ptr += tile_n_offset * K + tile_k_offset
        output_ptr += tile_n_offset * K + tile_k_offset
    else:
        tile_k_offset = pid_k * TILE_K
        out_grad_ptr += tile_k_offset
        in_grad_ptr += tile_k_offset
        output_ptr += tile_k_offset

    offs_m = pid_m * TILE_N + tl.arange(0, TILE_N)
    offs_n = tl.arange(0, TILE_K)
    idx = offs_m[:, None] * K + offs_n[None, :]

    mask = offs_n[None, :] < K - tile_k_offset

    logits = tl.load(output_ptr + idx, mask=mask, other=-float("inf")).to(tl.float32)
    logits -= tl.max(logits, axis=1)[:, None]
    numerator = tl.exp(logits)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = numerator / denominator

    out_grad = tl.load(out_grad_ptr + idx, mask=mask).to(tl.float32)
    in_grad = softmax_output * (out_grad - tl.sum(softmax_output * out_grad, axis=1)[:, None])

    tl.store(in_grad_ptr + idx, in_grad, mask=mask)

@triton.heuristics(
    {
        "TILE_K": lambda args: triton.next_power_of_2(args["K"]),
        "TILE_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.jit
def softmax_backward_kernel_inner(
    out_grad_ptr,
    in_grad_ptr,
    output_ptr,
    M,
    N,
    K,
    TILE_K: tl.const
