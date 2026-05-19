import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS", "HAS_X1", "HAS_W1", "HAS_B1"],
)
@triton.heuristics(
    {
        "HAS_RESIDUAL": lambda args: args["RESIDUAL"] is not None,
        "STORE_RESIDUAL_OUT": lambda args: args["residual_out"] is not None,
        "IS_RMS_NORM": lambda args: args["variance_epsilon"] == 0,
        "HAS_BIAS": lambda args: args["B"] is not None,
        "HAS_X1": lambda args: args["X1"] is not None,
        "HAS_W1": lambda args: args["W1"] is not None,
        "HAS_B1": lambda args: args["B1"] is not None,
    }
)
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    RESIDUAL,  # pointer to the residual
    X1,
    W1,
    B1,
    Y1,
    ROWSCALE,
    SEEDS,
    DROPOUT_MASK,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_res_row,
    stride_res1_row,
    stride_x1_row,
    stride_y1_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    dropout_p,  # dropout probability
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_X1: tl.constexpr,
    HAS_W1: tl.constexpr,
    HAS_B1: tl.constexpr,
    HAS_ROWSCALE: tl.constexpr,
    HAS_DROPOUT: tl.constexpr,
    STORE_DROPOUT_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if HAS_X1:
        X1 += row * stride_x1_row
    if STORE_RESIDUAL_OUT:
        residual_out = tl.zeros([BLOCK_N], dtype=tl.float32)
    if HAS_DROPOUT:
        keep_mask = tl.rand(tl.load(SEEDS + row)) > dropout_p
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_ROWSCALE:
        rowscale = tl.load(ROWSCALE + row).to(tl.float32)
        x *= rowscale
    if HAS_DROPOUT:
        x = tl.where(keep_mask, x / (1.0 - dropout_p), 0.0)
        if STORE_DROPOUT_MASK:
            tl.store(DROPOUT_MASK + row * N + cols, keep_mask, mask=cols < N)
    if HAS_X1:
        x1 = tl.load(X1 + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + M + row).to(tl.float32)
            x1 *= rowscale
        if HAS_DROPOUT:
            x1 = tl.where(keep_mask, x1 / (1.0 - dropout_p), 0.0)
            if STORE_DROPOUT_MASK:
                tl.store(DROPOUT_MASK + (M + row) * N + cols, keep_mask, mask=cols < N)
        x += x1
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + 2 * M + row).to(tl.float32)
            residual *= rowscale
        if HAS_DROPOUT:
            residual = tl.where(keep_mask, residual / (1.0 - dropout_p), 0.0)
            if STORE_DROPOUT_MASK:
                tl.store(DROPOUT_MASK + (2 * M + row) * N + cols, keep_mask, mask=cols < N)
        x += residual
        if STORE_RESIDUAL_OUT:
            residual_out += residual
    if STORE_RESIDUAL_OUT:
        tl.store(residual_out + cols, residual_out, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    tl.store(Y + cols, y, mask=mask)
    if HAS_W1:
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
        if HAS_B1:
            b1 = tl.load(B1 + cols, mask=mask).to(tl.float32)
        y1 = x_hat * w1 + b1 if HAS_B1 else x_hat * w1
        if HAS_DROPOUT:
            y1 = tl.where(keep_mask, y1 / (1.0 - dropout_p), 0.0)
            if STORE_DROPOUT_MASK:
                tl.store(DROPOUT_MASK + (M + row) * N + cols, keep_mask, mask=cols < N)
        tl.store(Y1 + cols, y1, mask=mask)


def _layer_norm_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    eps: float,
    residual: Optional[Tensor],
    x1: Optional[Tensor],
    weight1: Optional[Tensor],
    bias1: Optional[Tensor],
    rowscale: Optional[Tensor],
    dropout_p: float,
    seeds: Optional[Tensor],
    dropout_mask_store: Optional[Tensor],
    residual_out: Optional[Tensor],
    out: Optional[Tensor],
    out1: Optional[Tensor],
    is_rms_norm: bool
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    M, N = x.shape
    assert x.stride(-1) == 1
    if residual is not None:
        assert residual.stride(-1) == 1
        assert residual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if x1 is not None:
        assert x1.shape == x.shape
        assert rowscale is None
        assert x1.stride(-1) == 1
    if weight1 is not None:
        assert weight1.shape == (N,)
        assert weight1.stride(-1) == 1
    if bias1 is not None:
        assert bias1.shape == (N,)
        assert bias1.stride(-1) == 1
    if rowscale is not None:
        assert rowscale.is_contiguous()
        assert rowscale.shape == (M,)
    if seeds is not None:
        assert seeds.is_contiguous()
        assert seeds.shape == (M,)
    if dropout_mask_store is not None:
        assert dropout_mask_store.is_contiguous()
        assert dropout_mask_store.shape == (M, N) if x1 is None else (2 * M, N)
    if residual_out is not None:
        assert residual_out.is_contiguous()
        assert residual_out.shape == (M,
