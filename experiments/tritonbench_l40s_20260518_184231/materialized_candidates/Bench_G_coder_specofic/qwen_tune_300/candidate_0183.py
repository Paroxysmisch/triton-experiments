import torch
import triton
import triton.language as tl
from triton.language.extra import reduction

@triton.heuristics(
    {
        "TILE_K": lambda args: triton.next_power_of_2(args["K"] // args["num_warps"]),
        "TILE_N": lambda args: triton.next_power_of_2(args["N"] // 4),
        "ONE_TILE_PER_CTA": lambda args: args["TILE_N"] * args["TILE_K"] >= args["N"] * args["K"],
    }
)
@triton.jit
def softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """
    ========= ======== ======== =========
    M: batch  N: spatial K: feat  TILE_N:
    ========= ======== ======== =========
    """
    # Compute thread block index
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if ONE_TILE_PER_CTA:
        # Compute block start
        input_block_ptr = (
            input_ptr
            + (pid_m * N * K + pid_k * TILE_K) * tl.arange(0, TILE_N)[:, None] * K
        )
        output_block_ptr = (
            output_ptr
            + (pid_m * N * K + pid_k * TILE_K) * tl.arange(0, TILE_N)[:, None] * K
        )
        # Compute block
        src = tl.load(input_block_ptr, mask=tl.arange(0, TILE_N)[:, None] < N, other=-float("inf"))
        src = src.to(tl.float32)
        # Step 2: Compute linear layer
        exp_values = tl.exp(src - tl.max(src, axis=0))
        probabilities = exp_values / tl.sum(exp_values, axis=0)
        # Write back
        tl.store(output_block_ptr, probabilities.to(output_ptr.type.element_ty), mask=tl.arange(0, TILE_N)[:, None] < N)
    else:
        # Compute block start
        input_block_ptr = (
            input_ptr
            + (pid_m * N * K + pid_k * TILE_K) * tl.arange(0, TILE_N)[:, None] * K
        )
        output_block_ptr = (
            output_ptr
            + (pid_m * N * K + pid_k * TILE_K) * tl.arange(0, TILE_N)[:, None] * K
        )
        # Compute block
        src = tl.load(input_block_ptr, mask=(tl.arange(0, TILE_N)[:, None] < N) & (tl.arange(0, TILE_K)[None, :] < K), other=-float("inf"))
        src = src.to(tl.float32)
        # Step 2: Compute linear layer
        exp_values = tl.exp(src - tl.max(src, axis=0))
        probabilities = exp_values / tl.sum(exp_values, axis=0)
        # Write back
        tl.store(output_block_ptr, probabilities.to(output_ptr.type.element_ty), mask=(tl.arange(0, TILE_N)[:, None] < N) & (tl.arange(0, TILE_K)[None, :] < K))

@triton.heuristics(
    {
        "TILE_K": lambda args: triton.next_power_of_2(args["K"] // args["num_warps"]),
        "TILE_N": lambda args: triton.cdiv(args["N"], args["num_ctas"] * args["num_warps"]),
        "ONE_TILE_PER_CTA": lambda args: args["TILE_N"] == 1,
    }
)
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """
    ========= ======== ======== =========
    M: batch  N: spatial K: feat  TILE_N:
    ========= ======== ======== =========
    """
    # Compute thread block index
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if ONE_TILE_PER_CTA:
        # Compute block start
        input_block_ptr = input_ptr + (pid_m * K + pid_k * TILE_K) * N
        output_block_ptr = output_ptr + (pid_m * K + pid_k * TILE_K) * N
        # Compute block
        src = tl.load(input_block_ptr + tl.arange(0, TILE_N), mask=tl.arange(0, TILE_N) < N, other=-float("inf"))
        src = src.to(tl.float32)
        # Step 2: Compute linear layer
        exp_values = tl.exp(src - tl.max(src, axis=0))
        probabilities = exp_values / tl.sum(exp_values, axis=0)
        # Write back
        tl.store(output_block_ptr + tl.arange(0, TILE_N), probabilities.to(output_ptr.type.element_ty), mask=tl.arange(0, TILE_N) < N)
    else:
        # Compute block start
        input_block_ptr = input_ptr + (pid_m * K + pid_k * TILE_K) * N
        output_block_ptr = output_ptr + (pid_m * K + pid_k * TILE_K) * N
        # Compute block
        src = tl.load(input_block_ptr + tl.arange(0, TILE_N), mask=(tl.arange(0, TILE_N) < N) & (tl.arange(0, TILE_K) < K), other=-float("inf"))
        src = src.to(tl.float32)
        # Step 2: Compute linear layer
        exp_values = tl.exp(src - tl.max(src, axis=0))
        probabilities = exp_values / tl.sum(exp_values, axis=0)
        # Write back
        tl.store(output_block_ptr + tl.arange(0, TILE_N), probabilities.to(output_ptr.type.element_ty), mask=(tl.arange(0, TILE_N) < N) & (tl.arange(0, TILE_K) < K))

@triton.heuristics(
    {
        "TILE_K": lambda args: triton.next_power_of_2(args["K"] // args["num_warps"]),
        "TILE_N": lambda args: triton.next_power_of_2(args["N"] // 4),
        "ONE_TILE_PER_CTA": lambda args: args["TILE_N"] * args["TILE_K"] >= args["N"] * args["K"],
    }
)
@triton.jit
def softmax_backward_kernel_non_inner(
    output_ptr,
    in_grad_ptr,
    softmax_out_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    """
    ========= ======== ======== =========
    M: batch  N: spatial K: feat  TILE_N:
    ========= ======== ======== =========
    """
    # Compute thread block index
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if ONE_TILE_PER_CTA:
        # Compute block start
        in_grad_block_ptr = (
            in_grad_ptr
            + (pid_m * N * K + pid_k * TILE_K) * tl.arange(0, TILE_N)[:, None] * K
        )
        softmax_out_block_ptr = (
            softmax_out_ptr
            + (pid_m * N * K + pid_k * TILE_K) * tl.arange(0, TILE_N)[:, None] * K
        )
        input_block_ptr = (
            input_ptr
            + (pid_m * N * K + pid_k * TILE_K) * tl.arange(0, TILE_N)[:, None] * K
        )
        # Compute block
        src = tl.load(softmax_out_block_ptr, mask=tl.arange(0, TILE_N)[:, None] < N, other=0)
        src = src.to(tl.float32)
        input = tl.load(input_block_ptr, mask=tl.arange(0, TILE_N)[:, None] < N, other=-float("inf"))
        input = input.to(tl.float32)
        # Step 2: Compute linear layer
        src = src * (tl.exp(input - tl.max(input, axis=0)))
        grad_accum = tl.sum(src, axis=0)
        src = tl.load(
            output_ptr + (pid_m * N * K + pid_k * TILE_K) * tl.arange(0, TILE_N)[:, None] * K,
            mask=tl.arange(0, TILE_N)[:, None] < N,
            other=0,
        )
        src = src.to(tl.float32)
        tl.store(in_grad_block_ptr, (src * grad_accum - src * src) * tl.exp(input - tl.max(input, axis=0)), mask=tl.arange(0, TILE_N)[:, None] < N)
    else:
        # Compute block start
        in_grad_block_ptr = (
            in_grad_ptr
            + (pid_m * N * K + pid_k * TILE_K) * tl.arange(0, TILE_N)[:, None] * K
        )
        softmax_out_block_ptr = (
            softmax_out_ptr
            + (pid_m * N * K + pid_k * TILE_K) * tl.arange(0,
