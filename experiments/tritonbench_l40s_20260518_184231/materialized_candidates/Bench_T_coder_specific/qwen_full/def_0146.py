import torch
import triton
import triton.language as tl
from triton.language.math import exp2

@triton.jit
def elu_linear_fwd_fused(
    X,
    W,
    B,
    Y,
    alpha,
    stride_x_m,
    stride_x_k,
    stride_w_k,
    stride_w_n,
    stride_y_m,
    stride_y_n,
    n,
    m,
    k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    BIAS: tl.constexpr,
    INPLACE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_alpha = tl.program_id(1)
    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    # even_k = k % (BLOCK_K * GROUP_M) == 0
    # we only pre-compute some indices when we know k is divisible by BLOCK_K * GROUP_M
    if EVEN_K:
        rk = tl.arange(0, BLOCK_K)
        # the next lines are vectorized operations which means that we will get BLOCK_M * BLOCK_N * BLOCK_K results
        # in BLOCK_M * BLOCK_N blocks of size BLOCK_K. This is how we vectorize to improve performance
        x = tl.load(X + rm[:, None] * stride_x_m + rk[None, :] * stride_x_k)
        w = tl.load(W + rk[:, None] * stride_w_k + rn[None, :] * stride_w_n)
    else:
        rk = tl.arange(0, BLOCK_K * GROUP_M)
        # split the computation in GROUP_M smaller computation blocks of size k // GROUP_M
        x = tl.load(
            X + rm[:, None] * stride_x_m + (rk // GROUP_M) * stride_x_k,
            mask=rk < k,
            other=0,
        )
        w = tl.load(
            W + (rk[:, None] // GROUP_M) * stride_w_k + rn[None, :] * stride_w_n,
            mask=rk[:, None] < k,
            other=0,
        )
    # matmul
    y = tl.dot(x, w)
    # add bias
    if BIAS:
        b = tl.load(B + rn, mask=rn < n)
        y += b[None, :]
    # apply elu
    y = elu(y, alpha)
    # write back
    if INPLACE:
        tl.store(Y + rm[:, None] * stride_y_m + rn[None, :] * stride_y_n, y)
    else:
        tl.store(Y + rm[:, None] * stride_y_m + rn[None, :] * stride_y_n, y, mask=rm < m and rn < n)


def elu(x, alpha=1.0):
    return tl.where(x > 0, x, alpha * (exp2(x) - 1))


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "BATCH": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "BATCH": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8, "BATCH": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8, "BATCH": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "BATCH": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "BATCH": 16}, num_stages=2, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8, "BATCH": 16}, num_stages=2, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 2
