import triton
import triton.language as tl
import torch
import math
from typing import Tuple

@triton.jit
def _rms_layernorm_forward(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    IS_EVEN_N: tl.constexpr
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row

    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    mask = cols < N
    if IS_EVEN_N:
        w = tl.load(W + cols).to(tl.float32)
    else:
        w = tl.load(W + cols, mask=mask).to(tl.float32)

    x_hat = x * rstd
    y = x_hat * w

    # Write output
    if IS_EVEN_N:
        tl.store(Y + cols, y)
    else:
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def _rms_layernorm_backward(
    X,  # pointer to the input
    W,  # pointer to the weights
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DW,  # pointer to the partial sum of weights gradient
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_dy_row,
    stride_dx_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    rows_per_program,
    BLOCK_N: tl.constexpr,
    IS_EVEN_N: tl.constexpr
):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row

    w = tl.load(W + cols, mask=mask).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)

    row_end = min((row_block_id + 1) * rows_per_program, M)

    for row in range(row_start, row_end):
        if IS_EVEN_N:
            x = tl.load(X + cols).to(tl.float32)
            dy = tl.load(DY + cols).to(tl.float32)
        else:
            x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)

        rstd = tl.load(Rstd + row)

        x_hat = x * rstd
        if not IS_EVEN_N:
            x_hat = tl.where(mask, x_hat, 0.0)

        wdy = w * dy
        dw += dy * x_hat

        c1 = tl.sum(x_hat * wdy, axis=0) / N
        dx = (wdy - x_hat * c1) * rstd

        tl.store(DX + cols, dx, mask=mask)

        X += stride_x_row
        DY += stride_dy_row
        DX += stride_dx_row

    tl.store(DW + row_block_id * N + cols, dw, mask=mask)

@triton.jit
def _gemma_rms_layernorm_forward(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    IS_EVEN_N: tl.constexpr
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row

    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation with additional constant 1.0
    mask = cols < N
    if IS_EVEN_N:
        w = tl.load(W + cols).to(tl.float32)
    else:
        w = tl.load(W + cols, mask=mask).to(tl.float32)

    x_hat = x * rstd
    y = (x_hat * w) + 1.0

    # Write output
    if IS_EVEN_N:
        tl.store(Y + cols, y)
    else:
        tl.store(Y + cols, y, mask=mask)

import torch
import torch.nn as nn
from torch.autograd import Function

class Fast_RMS_Layernorm(Function):
    @staticmethod
    def forward(ctx, X, weight, eps):
        M, N = X.shape
        assert X.stride(-1) == 1
        assert weight.shape == (N,)
        assert weight.stride(-1) == 1

        # allocate output
        Y = torch.empty_like(X)
        rstd = torch.empty((M,), dtype=torch.float32, device=X.device)

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // X.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        assert N <= BLOCK_N

        # heuristics for number of warps
        with torch.cuda.device(X.device.index):
            _rms_layernorm_forward[(M,)](
                X,
                Y,
                weight,
                rstd,
                X.stride(0),
                Y.stride(0),
                N,
                eps,
                BLOCK_N,
                (N % BLOCK_N == 0)
            )

        ctx.save_for_backward(X, weight, rstd)
        ctx.eps = eps
        return Y

    @staticmethod
    def backward(ctx, dy):
        X, weight, rstd = ctx.saved_tensors
        eps = ctx.eps

        M, N = X.shape
        assert X.stride(-1) == 1
        assert dy.stride(-1) == 1
        assert dy.shape == (M, N)

        # allocate output
        dx = torch.empty_like(X)

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // X.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

        assert N <= BLOCK_N

        sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
        _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)

        rows_per_program = math.ceil(M / sm_count)
        grid = (sm_count,)
        with torch.cuda.device(X.device.index):
            _rms_layernorm_backward[grid](
                X,
                weight,
                dy,
                dx,
                _dw,
                rstd,
                X.stride(0),
                dy.stride(0),
                dx.stride(0),
                M,
                N,
                eps,
                rows_per_program,
                BLOCK_N,
                (N % BLOCK_N == 0)
            )
        dw = _dw.sum(0).to(weight.dtype)

        return dx, dw, None

def fast_rms_layernorm(X, weight, eps=1e-6):
    return Fast_RMS_Layernorm.apply(X, weight, eps)
