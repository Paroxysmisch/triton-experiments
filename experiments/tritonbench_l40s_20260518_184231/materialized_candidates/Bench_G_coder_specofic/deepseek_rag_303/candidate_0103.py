import torch
import triton
import triton.language as tl
import math
from torch import Tensor
from typing import Optional, Tuple

@triton.jit
def _rms_layernorm_forward(
    X, Y, W, Rstd, stride_x_row, stride_y_row, N, eps,
    BLOCK_SIZE: tl.constexpr, num_warps: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row

    cols = tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    xbar = tl.where(cols < N, x, 0.0)
    variance = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(variance + eps)

    mask = cols < N

    if num_warps == 1:
        w = tl.load(W + cols).to(tl.float32)
    else:
        w = tl.load(W + cols, mask=mask).to(tl.float32)

    y = (x * rstd) * w
    tl.store(Y + cols, y, mask=mask)

    tl.store(Rstd + row, rstd)

@triton.jit
def _rms_layernorm_backward(
    X, W, DY, DX, DW, Rstd, stride_x_row, stride_dy_row, stride_dx_row,
    M, N, eps, BLOCK_SIZE: tl.constexpr, num_warps: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x_row
    DY += row * stride_dy_row
    DX += row * stride_dx_row

    cols = tl.arange(0, BLOCK_SIZE)
    if num_warps > 1:
        mask = cols < N
    else:
        mask = None

    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    rstd = tl.load(Rstd + row)

    dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)

    xhat = x * rstd
    wdy = w * dy
    c1 = tl.sum(xhat * wdy, axis=0) / N
    dx = (wdy - xhat * c1) * rstd

    c2 = tl.sum(w * xhat, axis=0) / N
    dw = (wdy + xhat * c2) * rstd

    tl.store(DX + cols, dx, mask=mask)
    tl.store(DW + cols, dw, mask=mask)

@triton.jit
def _gemma_rms_layernorm_forward(
    X, Y, W, Rstd, stride_x_row, stride_y_row, N, eps, block_size,
    num_warps: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    # cols = tl.arange(0, BLOCK_SIZE)

    cols = tl.arange(0, block_size)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    xbar = tl.where(cols < N, x, 0.0)
    variance = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(variance + eps)
    tl.store(Rstd + row, rstd)

    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    xhat = x * rstd
    y = xhat * (w + CONSTANT_1_F32)

    tl.store(Y + cols, y, mask=mask)

class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def calculate_settings(N: int) -> Tuple[int, int]:
        num_warps = 8
        BLOCK_SIZE = triton.next_power_of_2(N)
        if BLOCK_SIZE > 2047:
            BLOCK_SIZE = 2048
        return BLOCK_SIZE, num_warps

    @staticmethod
    def fast_rms_layernorm(
        x: Tensor,
        weight: Tensor,
        eps: float,
        configuration: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        M, N = x.shape
        assert x.stride(-1) == 1

        assert weight.shape == (N,)
        assert weight.stride(-1) == 1

        BLOCK_SIZE, num_warps = Fast_RMS_Layernorm.calculate_settings(N)

        y = torch.empty_like(x)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        with torch.cuda.device(x.device.index):
            if configuration == "gemma":
                _gemma_rms_layernorm_forward[(M,)](
                    x,
                    y,
                    weight,
                    rstd,
                    x.stride(0),
                    y.stride(0),
                    N,
                    eps,
                    BLOCK_SIZE,
                    num_warps=num_warps,
                )
            else:
                _rms_layernorm_forward[(M,)](
                    x,
                    y,
                    weight,
                    rstd,
                    x.stride(0),
                    y.stride(0),
                    N,
                    eps,
                    BLOCK_SIZE,
                    num_warps=num_warps,
                )

        return y, rstd

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        eps: float,
        configuration: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        y, rstd = Fast_RMS_Layernorm.fast_rms_layernorm(x, weight, eps, configuration)
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        ctx.configuration = configuration
        return y, weight

    @staticmethod
    def backward(
        ctx,
        dy: Tensor,
        _: Tensor,
    ) -> Tuple[Tensor, None, None, None, None]:
        (x, weight, rstd) = ctx.saved_tensors
        M, N = x.shape

        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        BLOCK_SIZE, num_warps = Fast_RMS_Layernorm.calculate_settings(N)

        with torch.cuda.device(x.device.index):
            _rms_layernorm_backward[(M,)](
                x,
                weight,
                dy,
                dx,
                dw,
                rstd,
                x.stride(0),
                dy.stride(0),
                dx.stride(0),
                M,
                N,
                ctx.eps,
                BLOCK_SIZE,
                num_warps=num_warps,
            )
        return dx
