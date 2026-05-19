import triton
import triton.language as tl

@triton.jit
def rms_layernorm_forward_kernel(
    X,  # input tensor
    Y,  # output tensor
    W,  # weight tensor
    B,  # bias tensor
    Mean,  # mean tensor
    Rstd,  # reciprocal standard deviation tensor
    stride_xm,  # stride of input tensor in the feature dimension
    stride_ym,  # stride of output tensor in the feature dimension
    stride_wm,  # stride of weight tensor in the feature dimension
    stride_bm,  # stride of bias tensor in the feature dimension
    N,  # number of features
    eps,  # small epsilon value for numerical stability
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X + offsets, mask=mask)
    x_mean = tl.sum(x, axis=0) / N
    x_var = tl.sum((x - x_mean) * (x - x_mean), axis=0) / N
    x_rstd = 1.0 / tl.sqrt(x_var + eps)

    y = (x - x_mean) * x_rstd
    w = tl.load(W + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)
    y = y * w + b

    tl.store(Y + offsets, y, mask=mask)
    tl.store(Mean + pid, x_mean)
    tl.store(Rstd + pid, x_rstd)

@triton.jit
def rms_layernorm_backward_kernel(
    dY,  # gradient of output tensor
    X,  # input tensor
    W,  # weight tensor
    B,  # bias tensor
    Mean,  # mean tensor
    Rstd,  # reciprocal standard deviation tensor
    dX,  # gradient of input tensor
    dW,  # gradient of weight tensor
    dB,  # gradient of bias tensor
    stride_xm,  # stride of input tensor in the feature dimension
    stride_ym,  # stride of output tensor in the feature dimension
    stride_wm,  # stride of weight tensor in the feature dimension
    stride_bm,  # stride of bias tensor in the feature dimension
    N,  # number of features
    eps,  # small epsilon value for numerical stability
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X + offsets, mask=mask)
    dy = tl.load(dY + offsets, mask=mask)
    w = tl.load(W + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)
    mean = tl.load(Mean + pid)
    rstd = tl.load(Rstd + pid)

    x_centered = x - mean
    dy_centered = dy * w * rstd

    dvar = -0.5 * rstd * rstd * rstd * tl.sum(dy_centered * x_centered, axis=0)
    dmean = -rstd * tl.sum(dy_centered, axis=0) - 2.0 * dvar * mean / N

    dx = dy_centered * rstd + dvar * 2.0 * x_centered / N + dmean / N
    dw = tl.sum(dy * x_centered * rstd, axis=0)
    db = tl.sum(dy, axis=0)

    tl.store(dX + offsets, dx, mask=mask)
    tl.store(dW + offsets, dw, mask=mask)
    tl.store(dB + offsets, db, mask=mask)

import torch
from torch.autograd import Function

class FastRMSLayernorm(Function):
    @staticmethod
    def forward(ctx, x, w, b, eps=1e-5):
        assert x.is_cuda and w.is_cuda and b.is_cuda, "Triton kernels only support CUDA tensors"
        assert x.dim() == 2, "Input tensor must be 2D (batch_size, features)"
        assert w.dim() == 1 and b.dim() == 1, "Weight and bias tensors must be 1D"
        assert x.size(1) == w.size(0) == b.size(0), "Input tensor and weight/bias tensors must have the same number of features"

        batch_size, N = x.size()
        y = torch.empty_like(x)
        mean = torch.empty((batch_size,), device=x.device, dtype=x.dtype)
        rstd = torch.empty((batch_size,), device=x.device, dtype=x.dtype)

        grid = (batch_size,)
        BLOCK_SIZE = 256

        rms_layernorm_forward_kernel[grid](
            x, y, w, b, mean, rstd,
            x.stride(1), y.stride(1), w.stride(0), b.stride(0),
            N, eps, BLOCK_SIZE
        )

        ctx.save_for_backward(x, w, b, mean, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, w, b, mean, rstd = ctx.saved_tensors
        eps = ctx.eps

        batch_size, N = x.size()
        grad_input = torch.empty_like(x)
        grad_weight = torch.empty_like(w)
        grad_bias = torch.empty_like(b)

        grid = (batch_size,)
        BLOCK_SIZE = 256

        rms_layernorm_backward_kernel[grid](
            grad_output, x, w, b, mean, rstd,
            grad_input, grad_weight, grad_bias,
            x.stride(1), grad_output.stride(1), w.stride(0), b.stride(0),
            N, eps, BLOCK_SIZE
        )

        return grad_input, grad_weight, grad_bias, None

def fast_rms_layernorm(x, w, b, eps=1e-5):
    return FastRMSLayernorm.apply(x, w, b, eps)

import torch.nn as nn

class RMSLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(RMSLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return fast_rms_layernorm(x, self.weight, self.bias, self.eps)

# Example usage
if __name__ == "__main__":
    batch_size = 32
    features = 128
    x = torch.randn(batch_size, features, device='cuda')
    layernorm = RMSLayerNorm(features).to('cuda')

    y = layernorm(x)
    print(y)
