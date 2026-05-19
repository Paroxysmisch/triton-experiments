import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_forward_kernel(
    X, Y, W, B, Mean, Rstd,
    stride_x, stride_y,
    M, N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < M
    X_row = X + row * stride_x
    W_row = W + cols
    B_row = B + cols
    
    x = tl.load(X_row + cols, mask=mask, other=0.0)
    w = tl.load(W_row, mask=mask, other=0.0)
    b = tl.load(B_row, mask=mask, other=0.0)
    
    mean = tl.sum(x, axis=0) / M
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / M + eps
    rstd = 1.0 / tl.sqrt(var)
    
    y = x_centered * rstd * w + b
    tl.store(Y + row * stride_y + cols, y, mask=mask)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

@triton.jit
def _layer_norm_backward_kernel(
    X, DY, DX, DW, DB, Mean, Rstd, W,
    stride_x, stride_dy, stride_dx,
    M, N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < M
    X_row = X + row * stride_x
    DY_row = DY + row * stride_dy
    
    x = tl.load(X_row + cols, mask=mask, other=0.0)
    dy = tl.load(DY_row + cols, mask=mask, other=0.0)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    w = tl.load(W + cols, mask=mask, other=0.0)
    
    x_hat = (x - mean) * rstd
    dx_hat = dy * w
    
    sum1 = tl.sum(dx_hat * x_hat, axis=0)
    sum2 = tl.sum(dx_hat, axis=0)
    dx = (dx_hat - (x_hat * sum1 + sum2) / M) * rstd
    
    tl.store(DX + row * stride_dx + cols, dx, mask=mask)
    tl.atomic_add(DW + cols, x_hat * dy, mask=mask)
    tl.atomic_add(DB + cols, dy, mask=mask)

def layer_norm_forward(x, weight, bias, eps=1e-5):
    assert x.dim() == 2
    N, M = x.shape
    y = torch.empty_like(x)
    mean = torch.empty(N, dtype=torch.float32, device=x.device)
    rstd = torch.empty(N, dtype=torch.float32, device=x.device)
    
    BLOCK_SIZE = triton.next_power_of_2(M)
    num_warps = 4 if BLOCK_SIZE <= 2048 else 8
    
    _layer_norm_forward_kernel[(N,)](
        x, y, weight, bias, mean, rstd,
        x.stride(0), y.stride(0),
        M, N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y, mean, rstd

def layer_norm_backward(dy, x, weight, bias, mean, rstd):
    N, M = x.shape
    dx = torch.empty_like(x)
    dw = torch.zeros_like(weight)
    db = torch.zeros_like(bias)
    
    BLOCK_SIZE = triton.next_power_of_2(M)
    num_warps = 4 if BLOCK_SIZE <= 2048 else 8
    
    _layer_norm_backward_kernel[(N,)](
        x, dy, dx, dw, db, mean, rstd, weight,
        x.stride(0), dy.stride(0), dx.stride(0),
        M, N, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return dx, dw, db

class LigerLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        y, mean, rstd = layer_norm_forward(x, weight, bias, eps)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        eps = ctx.eps
        dx, dw, db = layer_norm_backward(dy, x, weight, bias, mean, rstd)
        return dx, dw, db, None

def liger_layer_norm(x, weight, bias, eps=1e-5):
    return LigerLayerNormFunction.apply(x, weight, bias, eps)
