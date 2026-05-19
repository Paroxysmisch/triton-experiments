import triton
import triton.language as tl

@triton.jit
def _layer_norm_forward_kernel(X, W, B, Y, Mean, RSTD, stride_xm, stride_ym, stride_xn, stride_wn, stride_bn, stride_yn, N, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offsets = row_idx * stride_xm + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load inputs
    x = tl.load(X + offsets, mask=mask, other=0.0)
    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    b = tl.load(B + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N
    var = tl.sum((x - mean) ** 2, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # Normalize
    y = (x - mean) * rstd * w + b

    # Store results
    tl.store(Y + offsets, y, mask=mask)
    tl.store(Mean + row_idx, mean)
    tl.store(RSTD + row_idx, rstd)

@triton.jit
def _layer_norm_backward_kernel(DY, X, Mean, RSTD, DW, DB, DX, stride_dym, stride_xm, stride_mean, stride_rstd, stride_dxn, N, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offsets = row_idx * stride_xm + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load inputs and gradients
    dy = tl.load(DY + offsets, mask=mask, other=0.0)
    x = tl.load(X + offsets, mask=mask, other=0.0)
    mean = tl.load(Mean + row_idx)
    rstd = tl.load(RSTD + row_idx)

    # Compute gradients
    dx = (dy - tl.sum(dy, axis=0) / N - (x - mean) * tl.sum(dy * (x - mean), axis=0) * rstd ** 2 / N) * rstd
    dw = dy * (x - mean) * rstd
    db = dy

    # Store results
    tl.store(DX + offsets, dx, mask=mask)
    tl.atomic_add(DW + tl.arange(0, BLOCK_SIZE), dw, mask=mask)
    tl.atomic_add(DB + tl.arange(0, BLOCK_SIZE), db, mask=mask)

import torch

class LigerLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        N = x.shape[-1]
        BLOCK_SIZE, num_warps = calculate_settings(N)

        y = torch.empty_like(x)
        mean = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
        rstd = torch.empty_like(mean)

        _layer_norm_forward_kernel[(x.shape[0],)](x, weight, bias, y, mean, rstd, x.stride(0), y.stride(0), x.stride(1), weight.stride(0), bias.stride(0), y.stride(1), N, BLOCK_SIZE=BLOCK_SIZE)

        ctx.save_for_backward(x, weight, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, rstd = ctx.saved_tensors
        N = x.shape[-1]
        BLOCK_SIZE, num_warps = calculate_settings(N)

        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight)
        db = torch.zeros_like(weight)

        _layer_norm_backward_kernel[(x.shape[0],)](dy, x, mean, rstd, dw, db, dx, dy.stride(0), x.stride(0), mean.stride(0), rstd.stride(0), dx.stride(1), N, BLOCK_SIZE=BLOCK_SIZE)

        return dx, dw, db

def calculate_settings(N):
    # Placeholder function for determining BLOCK_SIZE and num_warps
    BLOCK_SIZE = min(1024, N)
    num_warps = 4
    return BLOCK_SIZE, num_warps
