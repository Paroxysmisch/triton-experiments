import triton
import triton.language as tl

# Constants
MAX_FUSED_SIZE = 1024

@triton.jit
def calculate_settings(n: tl.constexpr) -> tl.int32:
    block_size = min(MAX_FUSED_SIZE, n)
    num_warps = 4 if block_size >= 512 else 2
    return block_size, num_warps

@triton.jit
def layernorm_forward(X, W, b, Y, r, mu, M, N, eps: tl.float32, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X + pid * N + offsets, mask=mask, other=0.0)
    w = tl.load(W + offsets, mask=mask, other=0.0)
    b = tl.load(b + offsets, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / N
    tl.store(mu + pid, mean)

    # Compute variance
    var = tl.sum((x - mean) ** 2, axis=0) / N
    r_val = 1.0 / tl.sqrt(var + eps)
    tl.store(r + pid, r_val)

    # Normalize and scale/shift
    y = (x - mean) * r_val * w + b
    tl.store(Y + pid * N + offsets, y, mask=mask)

@triton.jit
def layernorm_backward(dY, X, W, r, mu, dX, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    dy = tl.load(dY + pid * N + offsets, mask=mask, other=0.0)
    x = tl.load(X + pid * N + offsets, mask=mask, other=0.0)
    w = tl.load(W + offsets, mask=mask, other=0.0)
    r_val = tl.load(r + pid)
    mean = tl.load(mu + pid)

    # Compute gradients
    d_var = tl.sum(dy * (x - mean) * r_val, axis=0) * (-0.5) * r_val**3
    d_mean = tl.sum(dy * -r_val, axis=0) + d_var * (-2.0) * (x - mean) / N
    d_x = dy * r_val + d_var * 2.0 * (x - mean) / N + d_mean / N

    # Apply scaling
    d_x *= w
    tl.store(dX + pid * N + offsets, d_x, mask=mask)

import torch
from torch.autograd import Function

class Fast_Layernorm(Function):
    @staticmethod
    def forward(ctx, X, W, b, eps):
        M, N = X.shape
        block_size, num_warps = calculate_settings(N)
        Y = torch.empty_like(X)
        r = torch.empty(M, device=X.device, dtype=X.dtype)
        mu = torch.empty(M, device=X.device, dtype=X.dtype)

        grid = (M,)
        layernorm_forward[grid](X, W, b, Y, r, mu, M, N, eps, BLOCK_SIZE=block_size, num_warps=num_warps)

        ctx.save_for_backward(X, W, r, mu)
        ctx.eps = eps
        return Y

    @staticmethod
    def backward(ctx, dY):
        X, W, r, mu = ctx.saved_tensors
        M, N = X.shape
        block_size, num_warps = calculate_settings(N)
        dX = torch.empty_like(X)

        grid = (M,)
        layernorm_backward[grid](dY, X, W, r, mu, dX, M, N, BLOCK_SIZE=block_size, num_warps=num_warps)

        return dX, None, None, None

def fast_layernorm(X, W, b, eps=1e-5):
    return Fast_Layernorm.apply(X, W, b, eps)

import torch
import torch.nn as nn

# Example input
X = torch.randn(32, 128, device='cuda')
W = torch.randn(128, device='cuda', requires_grad=True)
b = torch.randn(128, device='cuda', requires_grad=True)

# Apply custom layer normalization
Y = fast_layernorm(X, W, b)

# Example loss and backward pass
loss = Y.sum()
loss.backward()

print("Output:", Y)
print("Gradient of X:", X.grad)
print("Gradient of W:", W.grad)
print("Gradient of b:", b.grad)
