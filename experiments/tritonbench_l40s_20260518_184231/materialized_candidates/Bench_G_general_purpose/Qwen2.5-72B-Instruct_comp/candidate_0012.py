import triton
import triton.language as tl

@triton.jit
def _layer_norm_forward_kernel(
    X, W, B, Y, Mean, RSTD,
    stride_x_row, stride_x_col,
    stride_w_col, stride_b_col,
    stride_y_row, stride_y_col,
    stride_mean_row, stride_rstd_row,
    N, eps, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    # Load data
    x_ptrs = X + row * stride_x_row + cols * stride_x_col
    w_ptrs = W + cols * stride_w_col
    b_ptrs = B + cols * stride_b_col
    y_ptrs = Y + row * stride_y_row + cols * stride_y_col
    mean_ptr = Mean + row * stride_mean_row
    rstd_ptr = RSTD + row * stride_rstd_row

    # Load data into Triton
    x = tl.load(x_ptrs, mask=cols < N, other=0.0)
    w = tl.load(w_ptrs, mask=cols < N, other=0.0)
    b = tl.load(b_ptrs, mask=cols < N, other=0.0)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N
    var = tl.sum(tl.square(x - mean), axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply weight and bias
    y = (x - mean) * rstd * w + b

    # Store results
    tl.store(mean_ptr, mean)
    tl.store(rstd_ptr, rstd)
    tl.store(y_ptrs, y, mask=cols < N)

@triton.jit
def _layer_norm_backward_kernel(
    DX, DW, DB, X, W, B, Y, Mean, RSTD,
    stride_x_row, stride_x_col,
    stride_w_col, stride_b_col,
    stride_y_row, stride_y_col,
    stride_mean_row, stride_rstd_row,
    stride_dx_row, stride_dx_col,
    stride_dw_col, stride_db_col,
    N, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    # Load data
    x_ptrs = X + row * stride_x_row + cols * stride_x_col
    w_ptrs = W + cols * stride_w_col
    b_ptrs = B + cols * stride_b_col
    y_ptrs = Y + row * stride_y_row + cols * stride_y_col
    mean_ptr = Mean + row * stride_mean_row
    rstd_ptr = RSTD + row * stride_rstd_row
    dx_ptrs = DX + row * stride_dx_row + cols * stride_dx_col
    dw_ptrs = DW + cols * stride_dw_col
    db_ptrs = DB + cols * stride_db_col

    # Load data into Triton
    x = tl.load(x_ptrs, mask=cols < N, other=0.0)
    w = tl.load(w_ptrs, mask=cols < N, other=0.0)
    b = tl.load(b_ptrs, mask=cols < N, other=0.0)
    y = tl.load(y_ptrs, mask=cols < N, other=0.0)
    mean = tl.load(mean_ptr)
    rstd = tl.load(rstd_ptr)

    # Compute gradients
    dy = tl.load(dx_ptrs, mask=cols < N, other=0.0)
    x_hat = (x - mean) * rstd
    d_x_hat = dy * w
    d_var = tl.sum(d_x_hat * (x - mean) * -0.5 * rstd * rstd * rstd, axis=0)
    d_mean = tl.sum(d_x_hat * -rstd, axis=0) + d_var * tl.sum(-2.0 * (x - mean), axis=0) / N
    dx = d_x_hat * rstd + d_var * 2.0 * (x - mean) / N + d_mean / N
    dw = tl.sum(dy * x_hat, axis=0)
    db = tl.sum(dy, axis=0)

    # Store results
    tl.atomic_add(dw_ptrs, dw, mask=cols < N)
    tl.atomic_add(db_ptrs, db, mask=cols < N)
    tl.store(dx_ptrs, dx, mask=cols < N)

import torch
from torch.autograd import Function

class LigerLayerNormFunction(Function):
    @staticmethod
    def forward(ctx, X, W, B, eps=1e-5):
        N, D = X.shape
        Y = torch.empty_like(X)
        Mean = torch.empty((N,), device=X.device, dtype=X.dtype)
        RSTD = torch.empty((N,), device=X.device, dtype=X.dtype)

        BLOCK_SIZE = 128
        num_warps = 4

        def calculate_settings(N, D):
            return BLOCK_SIZE, num_warps

        BLOCK_SIZE, num_warps = calculate_settings(N, D)

        _layer_norm_forward_kernel[(N,)](
            X, W, B, Y, Mean, RSTD,
            X.stride(0), X.stride(1),
            W.stride(0), B.stride(0),
            Y.stride(0), Y.stride(1),
            Mean.stride(0), RSTD.stride(0),
            D, eps, BLOCK_SIZE
        )

        ctx.save_for_backward(X, W, B, Y, Mean, RSTD)
        ctx.eps = eps
        return Y

    @staticmethod
    def backward(ctx, dY):
        X, W, B, Y, Mean, RSTD = ctx.saved_tensors
        N, D = X.shape
        DX = torch.empty_like(X)
        DW = torch.empty_like(W)
        DB = torch.empty_like(B)

        BLOCK_SIZE = 128
        num_warps = 4

        def calculate_settings(N, D):
            return BLOCK_SIZE, num_warps

        BLOCK_SIZE, num_warps = calculate_settings(N, D)

        _layer_norm_backward_kernel[(N,)](
            DX, DW, DB, X, W, B, Y, Mean, RSTD,
            X.stride(0), X.stride(1),
            W.stride(0), B.stride(0),
            Y.stride(0), Y.stride(1),
            Mean.stride(0), RSTD.stride(0),
            DX.stride(0), DX.stride(1),
            DW.stride(0), DB.stride(0),
            D, BLOCK_SIZE
        )

        return DX, DW, DB, None

import torch

# Example usage
N, D = 1024, 512
X = torch.randn(N, D, device='cuda')
W = torch.randn(D, device='cuda', requires_grad=True)
B = torch.randn(D, device='cuda', requires_grad=True)

Y = LigerLayerNormFunction.apply(X, W, B)

# Compute loss and backprop
loss = Y.sum()
loss.backward()

print(W.grad)
print(B.grad)
