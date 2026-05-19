import torch
import triton
import triton.language as tl

@triton.jit
def _rms_layernorm_forward(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * stride
    X += pid * stride
    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x, axis=0) / N
    inv_var = tl.math.rsqrt(var + eps)
    w = tl.load(W + cols, mask, other=0.0)
    y = x * inv_var * w
    tl.store(Y + cols, y, mask=mask)

@triton.jit
def _rms_layernorm_backward(
    X,  # pointer to the input
    W,  # pointer to the weights
    DX,  # pointer to the input gradient
    DW,  # pointer to the weight gradient
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    X += pid * stride
    DX += pid * stride
    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)
    w = tl.where(mask, w, 0.0)
    var = tl.sum(x * x, axis=0) / N
    inv_var = tl.math.rsqrt(var + eps)
    dw = tl.sum((x * inv_var) * w * (mask.to(tl.float32)), axis=0)
    dx = w * (inv_var * x) * (mask.to(tl.float32))
    tl.store(DX + cols, dx, mask=mask)
    tl.store(DW + cols, dw, mask=mask)

@triton.jit
def _gemma_rms_layernorm_forward(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * stride
    X += pid * stride
    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x, axis=0) / N
    inv_var = tl.math.rsqrt(var + eps)
    w = tl.load(W + cols, mask, other=0.0)
    y = x * inv_var * (w + 1.0)
    tl.store(Y + cols, y, mask=mask)

class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps):
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        num_warps = calculate_settings(N)[1]
        _rms_layernorm_forward[(M,)](
            x_arg,
            y,
            weight,
            x_arg.stride(0),
            N,
            eps,
            BLOCK_SIZE=1024,
            num_warps=num_warps,
        )
        ctx.save_for_backward(x, weight)
        ctx.BLOCK_SIZE = 1024
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        dx = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        num_warps = ctx.num_warps
        _rms_layernorm_backward[(M,)](
            x,
            w,
            dx,
            torch.empty_like(w),
            x_arg.stride(0),
            N,
            ctx.eps,
            BLOCK_SIZE=1024,
            num_warps=num_warps,
        )
        return dx, None, None, None

def fast_rms_layernorm(x, normalized_shape, weight, eps):
    return Fast_RMS_Layernorm.apply(x, normalized_shape, weight, eps)
