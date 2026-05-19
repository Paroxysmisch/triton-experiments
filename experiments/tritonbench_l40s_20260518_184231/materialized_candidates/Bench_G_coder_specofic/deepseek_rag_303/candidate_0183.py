import triton
import triton.language as tl

from torch._inductor.triton_heuristics import group_by_size
from torch._inductor.utils import instance_descriptor
from torch import empty_like

def heur_tile_k(args):
    return triton.next_power_of_2(args["K"])

def heur_tile_n_non_inner(args):
    return triton.next_power_of_2(max(32, args["N"] // args["K"]))

def heur_ctas_per_inbatch(args):
    return min(args["CTAS_PER_SM"], args["inbatch_ctas"])

def heur_tile_n_inner(args):
    return triton.next_power_of_2(max(32, args["N"]))

@instance_descriptor(
    lambda grouped_args: grouped_args,
    lambda key: (
        heur_tile_k,
        heur_ctas_per_inbatch,
        heur_tile_n_non_inner,
        heur_tile_n_non_inner,
        lambda args: args["num_warps"],
    )
    if key.endswith("non_inner")
    else (
        heur_tile_k,
        heur_ctas_per_inbatch,
        heur_tile_n_inner,
        heur_tile_n_inner,
        lambda args: args["num_warps"],
    ),
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
    pid = tl.program_id(axes=0)
    n_ctas = tl.num_programs(axes=0)
    cta_id = pid if ONE_TILE_PER_CTA else pid // TILE_K
    k_offs = cta_id * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        offs = tl.arange(0, TILE_N)
        x_ptrs = input_ptr + k_offs[:, None] * N + offs[None, :]
        row = tl.load(x_ptrs, mask=(offs[None, :] < N), other=-float("inf"))
    else:
        k_mask = k_offs < K
        offs = tl.arange(0, TILE_N) % TILE_N
        x_ptrs = input_ptr + k_offs[:, None] * N + offs[None, :]
        row = tl.load(x_ptrs, mask=(offs[None, :] < N), other=-float("inf"))

    if not ONE_TILE_PER_CTA:
        row_block_ptr = input_ptr + k_offs[:, None] * N
        row_mask = k_offs < K
    else:
        row_block_ptr = input_ptr
        row_mask = True

    if ONE_TILE_PER_CTA:
        row_minus_max = row - tl.max(row, axis=1)[:, None]
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=1)[:, None]
        softmax_out = numerator / denominator
        softmax_out_mask = (offs[None, :] < N) & row_mask

        o_ptrs = output_ptr + k_offs[:, None] * N + offs[None, :]
        tl.store(o_ptrs, softmax_out, mask=softmax_out_mask)
    else:
        row = tl.max(row, axis=1)[:, None]
        row_minus = row_block_ptr + offs[None, :]
        row_minus_max = row - tl.load(row_minus, mask=row_mask & (offs[None, :] < N))
        tl.store(row_minus, row_minus_max, mask=row_mask & (offs[None, :] < N))
        tl.debug_barrier()
        n_row = tl.sum(tl.exp(row_minus_max), axis=1)[:, None]
        tl.store(row_minus, n_row, mask=row_mask & (offs[None, :] < N))
        tl.debug_barrier()

        o_ptrs = output_ptr + k_offs[:, None] * N + offs[None, :]
        x = tl.load(o_ptrs, mask=(row_mask & (offs[None, :] < N)))
        softmax_out = tl.exp(x - row)
        tl.store(o_ptrs, softmax_out, mask=row_mask & (offs[None, :] < N))

def grad(softmax_out, softmax_in, grad_out):
    ds = softmax_out * (softmax_in - softmax_out)
    grad_in = ds * grad_out
    return grad_in

@instance_descriptor(
    lambda grouped_args: grouped_args,
    lambda key: (
        heur_tile_k,
        heur_ctas_per_inbatch,
        heur_tile_n_non_inner,
        heur_tile_n_non_inner,
        lambda args: args["num_warps"],
    )
    if key.endswith("non_inner")
    else (
        heur_tile_k,
        heur_ctas_per_inbatch,
        heur_tile_n_non_inner,
        heur_tile_n_non_inner,
        lambda args: args["num_warps"],
    ),
)
@triton.jit
def softmax_backward_kernel_non_inner(
    out_grad_ptr,
    in_grad_ptr,
    out_ptr,
    M,
    N,
    K,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid = tl.program_id(axes=0)
    n_ctas = tl.num_programs(axes=0)
    cta_id = pid if ONE_TILE_PER_CTA else pid // TILE_K
    k_offs = cta_id * TILE_K + tl.arange(0, TILE_K)

    # load in partial order
    if ONE_TILE_PER_CTA:
        offs = tl.arange(0, TILE_N)
        gg_ptrs = out_grad_ptr + k_offs[:, None] * N + offs[None, :]
        gg = tl.load(gg_ptrs, mask=(offs[None, :] < N))
        fg_ptrs = out_ptr + k_offs[:, None] * N + offs[None, :]
        fg = tl.load(fg_ptrs, mask=(offs[None, :] < N))
    else:
        gg_offs = tl.arange(0, TILE_K) * TILE_N
        gg_vals = tl.load(
            out_grad_ptr
            + (gg_offs[:, None] + tl.arange(0, TILE_N)[None, :]) * M
            + k_offs[:, None] * N,
        )
        fg_vals = tl.load(
            out_ptr
            + (gg_offs[:, None] + tl.arange(0, TILE_N)[None, :]) * M
            + k_offs[:, None] * N
