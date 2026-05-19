import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X, Y, Rstd, stride_x, stride_y, N, eps, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    base_idx = row * stride_x

    # Initialize sum of squares
    sum_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + base_idx + cols, mask=mask, other=0.0).to(tl.float32)
        sum_squares += x * x
    
    variance = tl.sum(sum_squares) / N
    rstd = 1.0 / tl.sqrt(variance + eps)
    
    # Store the reciprocal of the standard deviation
    tl.store(Rstd + row, rstd)
    
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + base_idx + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd
        tl.store(Y + row * stride_y + cols, y, mask=mask)

def _l2_norm_fwd(X, eps=1e-5, BLOCK_SIZE=1024):
    M, N = X.shape
    Y = torch.empty_like(X)
    Rstd = torch.empty(M, dtype=torch.float32, device=X.device)
    
    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](
        X, Y, Rstd, X.stride(0), Y.stride(0), N, eps, BLOCK_SIZE
    )
    
    return Y, Rstd

@triton.jit
def _l2_norm_bwd_kernel(
    X, DY, DX, Rstd, stride_x, stride_dy, stride_dx, N, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    base_idx = row * stride_x

    rstd = tl.load(Rstd + row)
    
    sum_grad = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + base_idx + cols, mask=mask, other=0.0).to(tl.float32)
        sum_grad += dy * x

    sum_grad = tl.sum(sum_grad)
    
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + base_idx + cols, mask=mask, other=0.0).to(tl.float32)
        dx = (dy - x * sum_grad / N) * rstd
        tl.store(DX + row * stride_dx + cols, dx, mask=mask)

def _l2_norm_bwd(X, DY, Rstd, BLOCK_SIZE=1024):
    M, N = X.shape
    DX = torch.empty_like(X)
    
    grid = (M,)
    _l2_norm_bwd_kernel[grid](
        X, DY, DX, Rstd, X.stride(0), DY.stride(0), DX.stride(0), N, BLOCK_SIZE
    )
    
    return DX

import torch

# Example usage
X = torch.randn(32, 1024, device='cuda')
Y, Rstd = _l2_norm_fwd(X)
DY = torch.randn_like(Y)
DX = _l2_norm_bwd(X, DY, Rstd)
