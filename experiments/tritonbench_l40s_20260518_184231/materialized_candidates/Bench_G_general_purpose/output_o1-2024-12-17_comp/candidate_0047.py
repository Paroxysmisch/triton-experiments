import math
import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 4096

def calculate_settings(n):
    BLOCK = 1
    while (BLOCK * 2) <= n and (BLOCK * 2) <= MAX_FUSED_SIZE:
        BLOCK *= 2
    # Choose number of warps based on block size
    WARPS = max(1, BLOCK // 256)
    return BLOCK, WARPS

@triton.jit
def layernorm_forward(
    X, W, B, R, MU,
    M, N, EPS, 
    stride_xm, stride_xn,
    stride_w, stride_b,
    BLOCK: tl.constexpr, 
):
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK)
    x_ptrs = X + pid_m * stride_xm + offs_n * stride_xn
    mask = offs_n < N
    x = tl.where(mask, tl.load(x_ptrs), 0.0)
    mean = tl.sum(x, axis=0) / N
    var = tl.sum((x - mean) * (x - mean), axis=0) / N
    r = 1.0 / tl.sqrt(var + EPS)
    w = tl.load(W + offs_n * stride_w, mask=mask).to(x.dtype)
    b = tl.load(B + offs_n * stride_b, mask=mask).to(x.dtype)
    y = (x - mean) * r * w + b
    tl.store(X + pid_m * stride_xm + offs_n * stride_xn, y, mask=mask)
    tl.store(R + pid_m * N + offs_n, r, mask=mask)
    if offs_n == 0:
        tl.store(MU + pid_m, mean)

@triton.jit
def layernorm_backward(
    DY, X, W, R, MU, DX,
    M, N, 
    stride_dym, stride_dyn,
    stride_w, stride_r,
    stride_dx_m, stride_dx_n,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK)
    mask = offs_n < N
    dy_ptrs = DY + pid_m * stride_dym + offs_n * stride_dyn
    dy = tl.where(mask, tl.load(dy_ptrs), 0.0)
    r_ptrs = R + pid_m * N + offs_n
    r = tl.where(mask, tl.load(r_ptrs), 0.0)
    w_ptrs = W + offs_n * stride_w
    w = tl.where(mask, tl.load(w_ptrs), 0.0)
    mu = tl.load(MU + pid_m)
    x_ptrs = X + pid_m * stride_dym + offs_n * stride_dyn
    x = tl.where(mask, tl.load(x_ptrs), 0.0)

    d_norm = dy * w
    d_mean = tl.sum(d_norm, axis=0) * (-r / N)
    d_r = tl.sum((x - mu) * d_norm, axis=0) * (-r**3 / N)
    dx = d_norm * r + (x - mu) * d_r + d_mean

    dx_ptrs = DX + pid_m * stride_dx_m + offs_n * stride_dx_n
    tl.store(dx_ptrs, dx, mask=mask)

class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, eps):
        M, N = X.shape
        BLOCK, WARPS = calculate_settings(N)
        X_ptr = X.contiguous()
        R = torch.empty_like(X)
        MU = torch.empty(M, device=X.device, dtype=X.dtype)
        grid = (M,)
        layernorm_forward[grid](
            X_ptr, W, b, R, MU,
            M, N, eps,
            X_ptr.stride(0), X_ptr.stride(1),
            W.stride(0), b.stride(0),
            BLOCK=BLOCK
        )
        ctx.save_for_backward(X_ptr, W, R, MU)
        ctx.N = N
        ctx.BLOCK = BLOCK
        return X_ptr

    @staticmethod
    def backward(ctx, dY):
        X, W, R, MU = ctx.saved_tensors
        M, N = X.shape
        DX = torch.empty_like(X)
        grid = (M,)
        layernorm_backward[grid](
            dY, X, W, R, MU, DX,
            M, N,
            dY.stride(0), dY.stride(1),
            W.stride(0), R.stride(1) if R.dim() > 1 else 1,
            DX.stride(0), DX.stride(1),
            BLOCK=ctx.BLOCK
        )
        dW = torch.sum((dY * (X - MU.unsqueeze(1)) * R), dim=0)
        dB = torch.sum(dY, dim=0)
        return DX, dW, dB, None

def fast_layernorm(x, ln_module):
    W = ln_module.weight
    b = ln_module.bias
    eps = ln_module.eps
    return Fast_Layernorm.apply(x, W, b, eps)
