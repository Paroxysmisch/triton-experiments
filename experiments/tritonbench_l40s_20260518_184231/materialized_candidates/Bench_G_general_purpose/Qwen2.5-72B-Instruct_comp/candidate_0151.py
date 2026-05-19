import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X, Y, W, B, RESIDUAL, X1, W1, B1, ROWSCALE, DROPOUT_MASK, SEEDS,
    Y1, residual_out, mean, invstd, N, eps, p, use_rms_norm: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load data
    x = tl.load(X + offsets, mask=mask, other=0.0)
    if RESIDUAL is not None:
        residual = tl.load(RESIDUAL + offsets, mask=mask, other=0.0)
        x += residual
    if X1 is not None:
        x1 = tl.load(X1 + offsets, mask=mask, other=0.0)
        x += x1
    if ROWSCALE is not None:
        rowscale = tl.load(ROWSCALE + pid, mask=mask, other=1.0)
        x *= rowscale

    # Compute mean and variance (or RMS norm)
    if use_rms_norm:
        var = tl.sum(x * x, axis=0) / N
        invstd = 1 / tl.sqrt(var + eps)
        mean = tl.zeros_like(var)
    else:
        mean = tl.sum(x, axis=0) / N
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered, axis=0) / N
        invstd = 1 / tl.sqrt(var + eps)

    # Apply weights and biases
    w = tl.load(W + offsets, mask=mask, other=1.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    y = (x - mean) * invstd * w + b

    # Apply dropout
    if p > 0:
        seeds = tl.load(SEEDS + pid, mask=mask, other=0)
        dropout_mask = tl.rand(seeds, offsets) < p
        y = tl.where(dropout_mask, 0.0, y / p)
        tl.store(DROPOUT_MASK + offsets, dropout_mask, mask=mask)

    # Store results
    tl.store(Y + offsets, y, mask=mask)
    if Y1 is not None:
        y1 = (x1 - mean) * invstd * w + b
        tl.store(Y1 + offsets, y1, mask=mask)
    if residual_out is not None:
        tl.store(residual_out + offsets, x, mask=mask)
    if mean is not None:
        tl.store(mean + pid, mean, mask=mask)
    if invstd is not None:
        tl.store(invstd + pid, invstd, mask=mask)

### Python Wrapper

import torch
import triton
import triton.language as tl

def layer_norm_fwd_1pass(X, Y, W, B, RESIDUAL=None, X1=None, W1=None, B1=None, ROWSCALE=None, DROPOUT_MASK=None, SEEDS=None, Y1=None, residual_out=None, mean=None, invstd=None, eps=1e-5, p=0.0, use_rms_norm=False):
    M, N = X.shape
    BLOCK_SIZE = 256

    # Allocate output tensors if not provided
    if Y is None:
        Y = torch.empty_like(X)
    if Y1 is not None:
        Y1 = torch.empty_like(X1)
    if residual_out is not None:
        residual_out = torch.empty_like(X)
    if mean is not None:
        mean = torch.empty((M,), device=X.device, dtype=X.dtype)
    if invstd is not None:
        invstd = torch.empty((M,), device=X.device, dtype=X.dtype)
    if DROPOUT_MASK is not None:
        DROPOUT_MASK = torch.empty_like(X, dtype=torch.bool)

    # Launch kernel
    grid = (M,)
    _layer_norm_fwd_1pass_kernel[grid](
        X, Y, W, B, RESIDUAL, X1, W1, B1, ROWSCALE, DROPOUT_MASK, SEEDS,
        Y1, residual_out, mean, invstd, N, eps, p, use_rms_norm, BLOCK_SIZE
    )

    return Y, Y1, residual_out, mean, invstd, DROPOUT_MASK
