import torch
import triton
import triton.language as tl
from triton.runtime.jit import MAX_FUSED_SIZE

@triton.jit
def calculate_settings(n):
    BLOCK = 128
    num_warps = 4
    if n > MAX_FUSED_SIZE // 2:
        num_warps = 8
    if n > MAX_FUSED_SIZE:
        BLOCK = MAX_FUSED_SIZE // 2
    return BLOCK, num_warps

@triton.jit
def layernorm_forward(
    X, Y, W, b, M, V, stride, N, eps,
    BLOCK: tl.constexpr, num_warps: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    X = X + row * stride + group * N * stride
    Y = Y + row * stride + group * N * stride
    p_mean = M + row * N + group * N * n_rows
    p_var = V + row * N + group * N * n_rows

    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    x = tl.where(mask, x, 0.)

    mean = tl.sum(x, axis=0) / N
    x = tl.where(mask, x - mean, 0.)
    x_zm = tl.where(mask, x, 0.)
    tl.store(p_mean + cols, mean, mask=mask)

    x = tl.where(mask, x, 0.)
    x = tl.where(mask, x * x, 0.)
    var = tl.sum(x, axis=0) / N
    tl.store(p_var + cols, var, mask=mask)

    x_zm = tl.where(mask, x_zm, 0.)
    std = tl.sqrt(var + eps)
    inv_std = 1. / std

    x_hat = x_zm * inv_std
    w = tl.load(W + cols, mask=mask, other=0.)
    b = tl.load(b + cols, mask=mask, other=0.)
    y = x_hat * w + b
    tl.store(Y + cols, y, mask=mask)

@triton.jit
def layernorm_backward(
    X, dY, dX, W, b, M, V, stride, N, eps, r, mu,
    BLOCK: tl.constexpr, num_warps: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    X = X + row * stride + group * N * stride
    dY = dY + row * stride + group * N * stride
    dX = dX + row * stride + group * N * stride
    p_mean = M + row * N + group * N * n_rows
    p_var = V + row * N + group * N * n_rows

    w = tl.load(W + cols, mask=mask, other=0.)
    mean = tl.load(p_mean + cols, mask=mask, other=0.)
    var = tl.load(p_var + cols, mask=mask, other=0.)
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    dy = tl.load(dY + cols, mask=mask, other=0.).to(tl.float32)

    x_hat = (x - mean) * tl.math.rsqrt(var + eps)
    r = tl.sum(x_hat * dy, axis=0) / N
    tl.store(this.r + cols, r, mask=mask)
    mu = tl.sum(dy * w, axis=0) / N
    tl.store(this.mu + cols, mu, mask=mask)

    d_x_hat = dy * w
    d_var = tl.sum(x_hat * d_x_hat, axis=0)
    d_mean = -tl.sum(d_x_hat, axis=0) + d_var * (x - mean) * tl.math.rsqrt(var + eps)

    d_x = d_x_hat * tl.math.rsqrt(var + eps) + d_mean * 1. / N + d_var * 2. * (x - mean) * tl.math.rsqrt(var + eps) / N
    tl.store(dX + cols, d_x, mask=mask)

class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        shape = x.shape
        n_rows = 1
        for dim in shape[:-1]:
            n_rows *= dim
        shape = (-1, -1)
        x = x.view(shape)
        M, N = x.shape
        V = N * n_rows
        W = N

        y = torch.empty_like(x)
        mean = torch.empty((M, n_rows), dtype=torch.float32, device=x.device)
        var = torch.empty((M, n_rows), dtype=torch.float32, device=x.device)
        num_warps = min(max_num_warps(x), 8)
        BLOCK, num_warps = calculate_settings(N)

        layernorm_forward[(M, n_rows)](
            x, y, weight, bias, mean, var,
            x.stride(0), N, eps,
            BLOCK=BLOCK, num_warps=num_warps,
            # ensure compatibility with old triton versions
            grid=(M, n_rows)
        )
        ctx.eps = eps
        ctx.BLOCK = BLOCK
        ctx.num_warps = num_warps
        ctx.save_for_backward(x, weight, bias, mean, var)
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, mean, var = ctx.saved_tensors
        shape = dy.shape
        n_rows = 1
        for dim in shape[:-1]:
            n_rows *= dim
        shape = (-1, -1)
        dy = dy.view(shape)
        M, N = dy.shape
        V = N * n_rows
        W = N

        dweight = torch.empty_like(weight)
        dbias = torch.empty_like(bias)
        dx = torch.empty_like(dy)
        r = torch.empty((M, n_rows), dtype=torch.float32, device=dy.device)
        mu = torch.empty((M, n_rows), dtype=torch.float32, device=dy.device)

        layernorm_backward[(M, n_rows)](
            x, dy, dx, weight, bias, mean, var,
            dx.stride(0), N, ctx.eps, r, mu,
            ctx.BLOCK, ctx.num_warps,
            # ensure compatibility with old triton versions
            grid=(M, n_rows)
        )
        dx = dx.view(shape)
        dweight = dweight.view(weight.shape)
        dbias = dbias.view(bias.shape)
        return dx, dweight, dbias, None

def fast_layernorm(x, weight, bias, eps):
    return Fast_Layernorm.apply(x, weight, bias, eps)
