import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=16),
    ],
    key=['N', 'use_residual', 'store_residual', 'use_rms_norm', 'use_bias']
)
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X, W, B, R, Y, 
    N, 
    use_residual: tl.constexpr, 
    store_residual: tl.constexpr, 
    use_rms_norm: tl.constexpr, 
    use_bias: tl.constexpr, 
    eps: tl.float32, 
    BLOCK_N: tl.constexpr
):
    row = tl.program_id(0)
    col_offset = tl.arange(0, BLOCK_N)
    x_ptrs = X + row * N + col_offset
    y_ptrs = Y + row * N + col_offset
    r_ptrs = R + row * N + col_offset if use_residual else None
    b_ptrs = B + col_offset if use_bias else None

    # Load data
    x = tl.load(x_ptrs, mask=col_offset < N, other=0.0)
    if use_residual:
        r = tl.load(r_ptrs, mask=col_offset < N, other=0.0)
        x = x + r

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N
    if use_rms_norm:
        var = tl.sum((x - mean) ** 2, axis=0) / N
    else:
        var = tl.sum(x ** 2, axis=0) / N - mean ** 2

    # Normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std

    # Apply weights and biases
    w = tl.load(W + col_offset, mask=col_offset < N, other=0.0)
    if use_bias:
        b = tl.load(b_ptrs, mask=col_offset < N, other=0.0)
        y = x_norm * w + b
    else:
        y = x_norm * w

    # Store results
    tl.store(y_ptrs, y, mask=col_offset < N)
    if store_residual:
        tl.store(r_ptrs, x, mask=col_offset < N)

import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=16),
    ],
    key=['N', 'use_residual', 'store_residual_grad', 'use_rms_norm', 'use_bias']
)
@triton.jit
def _layer_norm_bwd_kernel(
    X, W, B, R, Y, dY, dX, dW, dB, dR, 
    N, 
    use_residual: tl.constexpr, 
    store_residual_grad: tl.constexpr, 
    use_rms_norm: tl.constexpr, 
    use_bias: tl.constexpr, 
    eps: tl.float32, 
    BLOCK_N: tl.constexpr
):
    row = tl.program_id(0)
    col_offset = tl.arange(0, BLOCK_N)
    x_ptrs = X + row * N + col_offset
    y_ptrs = Y + row * N + col_offset
    r_ptrs = R + row * N + col_offset if use_residual else None
    dy_ptrs = dY + row * N + col_offset
    dx_ptrs = dX + row * N + col_offset
    dr_ptrs = dR + row * N + col_offset if store_residual_grad else None
    b_ptrs = B + col_offset if use_bias else None
    dw_ptrs = dW + col_offset
    db_ptrs = dB + col_offset if use_bias else None

    # Load data
    x = tl.load(x_ptrs, mask=col_offset < N, other=0.0)
    if use_residual:
        r = tl.load(r_ptrs, mask=col_offset < N, other=0.0)
        x = x + r
    y = tl.load(y_ptrs, mask=col_offset < N, other=0.0)
    dy = tl.load(dy_ptrs, mask=col_offset < N, other=0.0)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N
    if use_rms_norm:
        var = tl.sum((x - mean) ** 2, axis=0) / N
    else:
        var = tl.sum(x ** 2, axis=0) / N - mean ** 2
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Compute gradients
    w = tl.load(W + col_offset, mask=col_offset < N, other=0.0)
    x_norm = (x - mean) * inv_std
    dy_w = dy * w
    d_mean = tl.sum(dy_w * (-inv_std), axis=0)
    d_var = tl.sum(dy_w * (x - mean) * (-0.5) * inv_std ** 3, axis=0)
    dx = dy_w * inv_std + d_mean * (1.0 / N) + d_var * 2.0 * (x - mean) * (1.0 / N)
    if use_bias:
        db = tl.sum(dy, axis=0)
        tl.store(db_ptrs, db, mask=col_offset < N)
    dw = tl.sum(dy * x_norm, axis=0)
    tl.store(dw_ptrs, dw, mask=col_offset < N)

    # Store results
    tl.store(dx_ptrs, dx, mask=col_offset < N)
    if store_residual_grad:
        dr = dx
        tl.store(dr_ptrs, dr, mask=col_offset < N)

import torch
import triton
import triton.language as tl

def layer_norm_fwd_1pass(X, W, B=None, R=None, use_residual=False, store_residual=False, use_rms_norm=False, use_bias=False, eps=1e-6):
    N = X.shape[1]
    Y = torch.empty_like(X)
    R_out = torch.empty_like(X) if store_residual else None

    _layer_norm_fwd_1pass_kernel[(X.shape[0],)](
        X, W, B, R, Y, 
        N, 
        use_residual, 
        store_residual, 
        use_rms_norm, 
        use_bias, 
        eps
    )

    return Y, R_out

def layer_norm_bwd_1pass(X, W, B=None, R=None, Y=None, dY, dX, dW, dB=None, dR=None, use_residual=False, store_residual_grad=False, use_rms_norm=False, use_bias=False, eps=1e-6):
    N = X.shape[1]
    _layer_norm_bwd_kernel[(X.shape[0],)](
        X, W, B, R, Y, dY, dX, dW, dB, dR, 
        N, 
        use_residual, 
        store_residual_grad, 
        use_rms_norm, 
        use_bias, 
        eps
    )

# Example usage
X = torch.randn(1024, 1024, device='cuda')
W = torch.randn(1024, device='cuda')
B = torch.randn(1024, device='cuda')
R = torch.randn(1024, device='cuda')

Y, R_out = layer_norm_fwd_1pass(X, W, B, R, use_residual=True, store_residual=True, use_rms_norm=False, use_bias=True, eps=1e-6)

dY = torch.randn_like(Y)
dX = torch.empty_like(X)
dW = torch.empty_like(W)
dB = torch.empty_like(B)
dR = torch.empty_like(R)

layer_norm_bwd_1pass(X, W, B, R, Y, dY, dX, dW, dB, dR, use_residual=True, store_residual_grad=True, use_rms_norm=False, use_bias=True, eps=1e-6)
