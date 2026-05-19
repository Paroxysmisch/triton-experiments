import triton
import triton.language as tl

@triton.jit
def layernorm_forward(
    X,  # input tensor
    Y,  # output tensor
    W,  # weight tensor
    B,  # bias tensor
    M,  # mean tensor
    V,  # variance tensor
    eps,  # epsilon
    stride,  # stride of the input tensor
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    x_ptrs = X + row * stride + cols
    w_ptrs = W + cols
    b_ptrs = B + cols

    # Load data
    x = tl.load(x_ptrs, mask=cols < stride, other=0.0)
    w = tl.load(w_ptrs, mask=cols < stride, other=0.0)
    b = tl.load(b_ptrs, mask=cols < stride, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / stride
    tl.store(M + row, mean)

    # Compute variance
    var = tl.sum(tl.sqr(x - mean), axis=0) / stride
    tl.store(V + row, var)

    # Normalize and scale
    rstd = 1.0 / tl.sqrt(var + eps)
    y = (x - mean) * rstd * w + b

    # Store output
    tl.store(Y + row * stride + cols, y, mask=cols < stride)

@triton.jit
def layernorm_backward(
    dY,  # gradient of output tensor
    X,  # input tensor
    W,  # weight tensor
    B,  # bias tensor
    M,  # mean tensor
    V,  # variance tensor
    dX,  # gradient of input tensor
    dW,  # gradient of weight tensor
    dB,  # gradient of bias tensor
    eps,  # epsilon
    stride,  # stride of the input tensor
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    x_ptrs = X + row * stride + cols
    w_ptrs = W + cols
    b_ptrs = B + cols
    dy_ptrs = dY + row * stride + cols
    dx_ptrs = dX + row * stride + cols

    # Load data
    x = tl.load(x_ptrs, mask=cols < stride, other=0.0)
    w = tl.load(w_ptrs, mask=cols < stride, other=0.0)
    dy = tl.load(dy_ptrs, mask=cols < stride, other=0.0)

    # Load mean and variance
    mean = tl.load(M + row)
    var = tl.load(V + row)
    rstd = 1.0 / tl.sqrt(var + eps)

    # Compute gradients
    dx = dy * w * rstd
    sum_dy = tl.sum(dy, axis=0)
    sum_dy_x = tl.sum(dy * (x - mean), axis=0)
    dx -= (sum_dy * (x - mean) * rstd + sum_dy_x * rstd * rstd * (x - mean)) / stride
    dx *= rstd

    # Compute dW and dB
    dW += tl.sum(dy * (x - mean) * rstd, axis=0)
    dB += tl.sum(dy, axis=0)

    # Store gradients
    tl.store(dx_ptrs, dx, mask=cols < stride)
    tl.atomic_add(dW + cols, dW, mask=cols < stride)
    tl.atomic_add(dB + cols, dB, mask=cols < stride)

import torch
from torch.autograd import Function

class Fast_Layernorm(Function):
    @staticmethod
    def forward(ctx, X, W, B, eps=1e-5):
        M = torch.empty_like(X[:, :1])
        V = torch.empty_like(X[:, :1])
        Y = torch.empty_like(X)

        # Determine block size and number of warps
        stride = X.shape[1]
        BLOCK_SIZE = 128
        num_warps = 4

        # Launch Triton kernel
        layernorm_forward[(X.shape[0],)](X, Y, W, B, M, V, eps, stride, BLOCK_SIZE, num_warps=num_warps)

        ctx.save_for_backward(X, W, B, M, V, Y)
        ctx.eps = eps

        return Y

    @staticmethod
    def backward(ctx, dY):
        X, W, B, M, V, Y = ctx.saved_tensors
        eps = ctx.eps

        dX = torch.empty_like(X)
        dW = torch.zeros_like(W)
        dB = torch.zeros_like(B)

        # Determine block size and number of warps
        stride = X.shape[1]
        BLOCK_SIZE = 128
        num_warps = 4

        # Launch Triton kernel
        layernorm_backward[(X.shape[0],)](dY, X, W, B, M, V, dX, dW, dB, eps, stride, BLOCK_SIZE, num_warps=num_warps)

        return dX, dW, dB, None

# Utility function to apply layer normalization
def fast_layernorm(X, W, B, eps=1e-5):
    return Fast_Layernorm.apply(X, W, B, eps)

def calculate_settings(num_cols):
    if num_cols < 128:
        block_size = 32
        num_warps = 2
    elif num_cols < 256:
        block_size = 64
        num_warps = 4
    else:
        block_size = 128
        num_warps = 8
    return block_size, num_warps

import torch

# Create input tensor, weights, and biases
X = torch.randn(1024, 512, device='cuda')
W = torch.randn(512, device='cuda', requires_grad=True)
B = torch.randn(512, device='cuda', requires_grad=True)

# Apply layer normalization
Y = fast_layernorm(X, W, B)

# Compute loss and backpropagate
loss = Y.sum()
loss.backward()

print(W.grad)
print(B.grad)
