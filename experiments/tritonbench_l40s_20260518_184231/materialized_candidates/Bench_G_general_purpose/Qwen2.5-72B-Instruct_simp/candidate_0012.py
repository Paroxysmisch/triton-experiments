import triton
import triton.language as tl

@triton.jit
def _layer_norm_forward_kernel(X, Y, Mean, RSTD, W, B, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)
    mask = col_idx < n_cols

    # Load data
    x = tl.load(X + row_idx * n_cols + col_idx, mask=mask, other=0.0)
    w = tl.load(W + col_idx, mask=mask, other=0.0)
    b = tl.load(B + col_idx, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / n_cols
    tl.store(Mean + row_idx, mean)

    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(RSTD + row_idx, rstd)

    # Normalize and scale
    y = (x_centered * rstd) * w + b
    tl.store(Y + row_idx * n_cols + col_idx, y, mask=mask)

@triton.jit
def _layer_norm_backward_kernel(dY, X, W, B, Mean, RSTD, dX, dW, dB, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)
    mask = col_idx < n_cols

    # Load data
    dy = tl.load(dY + row_idx * n_cols + col_idx, mask=mask, other=0.0)
    x = tl.load(X + row_idx * n_cols + col_idx, mask=mask, other=0.0)
    w = tl.load(W + col_idx, mask=mask, other=0.0)
    b = tl.load(B + col_idx, mask=mask, other=0.0)
    mean = tl.load(Mean + row_idx)
    rstd = tl.load(RSTD + row_idx)

    # Compute x_centered
    x_centered = x - mean

    # Compute gradients
    d_var = tl.sum(dy * x_centered * w, axis=0) * (-0.5) * rstd * rstd * rstd
    d_mean = tl.sum(dy * w, axis=0) * (-rstd) + d_var * (-2.0) * tl.sum(x_centered, axis=0) / n_cols
    d_x = (dy * w * rstd) + (2.0 * d_var * x_centered / n_cols) + (d_mean / n_cols)
    d_w = tl.sum(dy * x_centered * rstd, axis=0)
    d_b = tl.sum(dy, axis=0)

    # Store gradients
    tl.store(dX + row_idx * n_cols + col_idx, d_x, mask=mask)
    tl.atomic_add(dW + col_idx, d_w, mask=mask)
    tl.atomic_add(dB + col_idx, d_b, mask=mask)

import torch

def layer_norm_forward(X, W, B, eps=1e-5):
    n_rows, n_cols = X.shape
    Y = torch.empty_like(X)
    Mean = torch.empty((n_rows,), device=X.device, dtype=X.dtype)
    RSTD = torch.empty((n_rows,), device=X.device, dtype=X.dtype)

    grid = (n_rows,)
    BLOCK_SIZE = 128
    num_warps = 4

    _layer_norm_forward_kernel[grid](X, Y, Mean, RSTD, W, B, n_cols, eps, BLOCK_SIZE, num_warps=num_warps)

    return Y, Mean, RSTD

def layer_norm_backward(dY, X, W, B, Mean, RSTD, eps=1e-5):
    n_rows, n_cols = X.shape
    dX = torch.empty_like(X)
    dW = torch.zeros_like(W)
    dB = torch.zeros_like(B)

    grid = (n_rows,)
    BLOCK_SIZE = 128
    num_warps = 4

    _layer_norm_backward_kernel[grid](dY, X, W, B, Mean, RSTD, dX, dW, dB, n_cols, BLOCK_SIZE, num_warps=num_warps)

    return dX, dW, dB

class LigerLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B, eps=1e-5):
        X = X.contiguous()
        W = W.contiguous()
        B = B.contiguous()

        Y, Mean, RSTD = layer_norm_forward(X, W, B, eps)
        ctx.save_for_backward(X, W, B, Mean, RSTD, eps)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X, W, B, Mean, RSTD, eps = ctx.saved_tensors
        dX, dW, dB = layer_norm_backward(grad_output, X, W, B, Mean, RSTD, eps)
        return dX, dW, dB, None

import torch.nn as nn

class LigerLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(LigerLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return LigerLayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# Example usage
x = torch.randn(32, 64, device='cuda')
ln = LigerLayerNorm(64).to('cuda')
y = ln(x)
y.backward(torch.randn_like(y))
