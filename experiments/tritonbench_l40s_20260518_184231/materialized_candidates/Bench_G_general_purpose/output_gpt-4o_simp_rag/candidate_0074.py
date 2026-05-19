import triton
import triton.language as tl
import torch

# Forward kernel
@triton.jit
def _layer_norm_fwd_fused(
    X, Y, W, B, Rstd, stride_m, stride_n, N, eps, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    base_idx = row_idx * stride_m
    X += base_idx
    Y += base_idx
    W += base_idx
    B += base_idx

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _mean += x
        _var += x * x

    mean = tl.sum(_mean) / N
    var = tl.sum(_var) / N - mean * mean
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row_idx, rstd)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        y = (x - mean) * rstd * w + b
        tl.store(Y + cols, y, mask=mask)

# Backward kernel for input gradient
@triton.jit
def _layer_norm_bwd_dx_fused(
    DY, X, W, Rstd, DX, stride_m, stride_n, N, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    base_idx = row_idx * stride_m
    DY += base_idx
    X += base_idx
    W += base_idx
    DX += base_idx

    rstd = tl.load(Rstd + row_idx)
    _sum_dy = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sum_dy_xhat = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        x_hat = (x - tl.sum(x) / N) * rstd
        _sum_dy += dy * w
        _sum_dy_xhat += dy * x_hat * w

    sum_dy = tl.sum(_sum_dy)
    sum_dy_xhat = tl.sum(_sum_dy_xhat)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        x_hat = (x - tl.sum(x) / N) * rstd
        dx = w * rstd * (dy - sum_dy / N - x_hat * sum_dy_xhat / N)
        tl.store(DX + cols, dx, mask=mask)

# Backward kernel for weight and bias gradient
@triton.jit
def _layer_norm_bwd_dwdb(
    DY, X, Rstd, DW, DB, stride_m, stride_n, N, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    base_idx = row_idx * stride_m
    DY += base_idx
    X += base_idx
    DW += base_idx
    DB += base_idx

    rstd = tl.load(Rstd + row_idx)
    _dw = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _db = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - tl.sum(x) / N) * rstd
        _dw += dy * x_hat
        _db += dy

    dw = tl.sum(_dw)
    db = tl.sum(_db)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        tl.atomic_add(DW + cols, dw, mask=mask)
        tl.atomic_add(DB + cols, db, mask=mask)

# PyTorch-compatible LayerNorm wrapper
class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        M, N = x.shape
        BLOCK_SIZE = 128  # Example block size
        y = torch.empty_like(x)
        rstd = torch.empty(M, device=x.device, dtype=x.dtype)

        _layer_norm_fwd_fused[(M,)](x, y, weight, bias, rstd, x.stride(0), x.stride(1), N, eps, BLOCK_SIZE)
        
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        M, N = x.shape
        BLOCK_SIZE = 128  # Example block size
        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight)
        db = torch.zeros_like(weight)

        _layer_norm_bwd_dx_fused[(M,)](dy, x, weight, rstd, dx, x.stride(0), x.stride(1), N, BLOCK_SIZE)
        _layer_norm_bwd_dwdb[(M,)](dy, x, rstd, dw, db, x.stride(0), x.stride(1), N, BLOCK_SIZE)

        return dx, dw, db, None

# Usage example
x = torch.randn(32, 128, device='cuda')
weight = torch.ones(128, device='cuda')
bias = torch.zeros(128, device='cuda')
eps = 1e-5

layer_norm = LayerNorm.apply
y = layer_norm(x, weight, bias, eps)
