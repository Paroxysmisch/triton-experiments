import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # input tensor
    W,  # weight tensor
    B,  # bias tensor
    Y,  # output tensor
    RES,  # residual tensor (optional)
    RMS,  # RMS normalization flag
    EPS,  # epsilon for numerical stability
    stride,  # stride of the input tensor
    N,  # number of elements in the last dimension
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load data
    x = tl.load(X + offsets, mask=mask, other=0.0)
    w = tl.load(W + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)

    # Compute mean and variance
    if RMS:
        var = tl.sum(x * x, axis=0) / N
        inv_std = 1.0 / tl.sqrt(var + EPS)
    else:
        mean = tl.sum(x, axis=0) / N
        var = tl.sum((x - mean) * (x - mean), axis=0) / N
        inv_std = 1.0 / tl.sqrt(var + EPS)

    # Normalize and apply weights and biases
    x_norm = (x - mean) * inv_std if not RMS else x * inv_std
    y = x_norm * w + b

    # Add residual connection if provided
    if RES is not None:
        res = tl.load(RES + offsets, mask=mask, other=0.0)
        y += res

    # Store the result
    tl.store(Y + offsets, y, mask=mask)

@triton.jit
def _layer_norm_bwd_kernel(
    dY,  # gradient of the output tensor
    X,  # input tensor
    W,  # weight tensor
    B,  # bias tensor
    DX,  # gradient of the input tensor
    DW,  # gradient of the weight tensor
    DB,  # gradient of the bias tensor
    RMS,  # RMS normalization flag
    EPS,  # epsilon for numerical stability
    stride,  # stride of the input tensor
    N,  # number of elements in the last dimension
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load data
    x = tl.load(X + offsets, mask=mask, other=0.0)
    w = tl.load(W + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    dy = tl.load(dY + offsets, mask=mask, other=0.0)

    # Compute mean and variance
    if RMS:
        var = tl.sum(x * x, axis=0) / N
        inv_std = 1.0 / tl.sqrt(var + EPS)
    else:
        mean = tl.sum(x, axis=0) / N
        var = tl.sum((x - mean) * (x - mean), axis=0) / N
        inv_std = 1.0 / tl.sqrt(var + EPS)

    # Normalize and apply weights and biases
    x_norm = (x - mean) * inv_std if not RMS else x * inv_std

    # Compute gradients
    dx_norm = dy * w
    dvar = tl.sum(dx_norm * (x - mean) * -0.5 * inv_std**3, axis=0)
    dmean = tl.sum(dx_norm * -inv_std, axis=0) + dvar * -2.0 * mean / N
    dx = dx_norm * inv_std + dvar * 2.0 * (x - mean) / N + dmean / N
    dw = tl.sum(dy * x_norm, axis=0)
    db = tl.sum(dy, axis=0)

    # Store the gradients
    tl.store(DX + offsets, dx, mask=mask)
    tl.store(DW + offsets, dw, mask=mask)
    tl.store(DB + offsets, db, mask=mask)

import torch

def layer_norm_fwd_1pass(x, w, b, res=None, rms=False, eps=1e-5):
    assert x.is_cuda and w.is_cuda and b.is_cuda
    assert x.shape[-1] == w.shape[0] == b.shape[0]
    N = x.shape[-1]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    y = torch.empty_like(x)
    _layer_norm_fwd_1pass_kernel[grid](
        x, w, b, y, res, rms, eps, x.stride(-1), N, BLOCK_SIZE
    )
    return y

def layer_norm_bwd_1pass(dy, x, w, b, rms=False, eps=1e-5):
    assert dy.is_cuda and x.is_cuda and w.is_cuda and b.is_cuda
    assert dy.shape == x.shape and x.shape[-1] == w.shape[0] == b.shape[0]
    N = x.shape[-1]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    dx = torch.empty_like(x)
    dw = torch.empty_like(w)
    db = torch.empty_like(b)
    _layer_norm_bwd_kernel[grid](
        dy, x, w, b, dx, dw, db, rms, eps, x.stride(-1), N, BLOCK_SIZE
    )
    return dx, dw, db
