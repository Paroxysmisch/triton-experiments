import triton
import triton.language as tl
import torch

@triton.jit
def _l2_norm_fwd_1pass_kernel(X, Y, variance, rstd, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    row_start = row * X.stride(0)
    col_start = col * X.stride(1)
    
    # Load data from X
    x = X[row_start + col_start]
    
    # Compute the sum of squares for the current row
    if col == 0:
        variance[row] = 0.0
    variance[row] = tl.atomic_add(variance[row], x * x)
    
    # Compute the reciprocal square root of the variance (rstd)
    if col == BLOCK_SIZE - 1:
        variance[row] /= BLOCK_SIZE
        rstd[row] = 1.0 / tl.sqrt(variance[row])
    
    # Normalize x and store the result in Y
    y = x * rstd[row]
    Y[row_start + col_start] = y

@triton.jit
def _l2_norm_bwd_kernel(X, DY, DX, variance, rstd, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    row_start = row * X.stride(0)
    col_start = col * X.stride(1)
    
    # Load inputs and gradients
    x = X[row_start + col_start]
    dy = DY[row_start + col_start]
    
    # Compute the gradient with respect to x
    dx = dy * rstd[row]
    dx -= x * dy * variance[row] * (1.0 / (variance[row] ** 2))
    
    # Store the computed gradient in DX
    DX[row_start + col_start] = dx

def l2_norm_fwd(X, BLOCK_SIZE=256):
    variance = torch.zeros((X.shape[0],), device=X.device, dtype=X.dtype)
    rstd = torch.zeros((X.shape[0],), device=X.device, dtype=X.dtype)
    
    Y = torch.zeros_like(X)
    
    grid = (X.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE, (X.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    _l2_norm_fwd_1pass_kernel[X.shape[0], BLOCK_SIZE](X, Y, variance, rstd, BLOCK_SIZE=BLOCK_SIZE)
    
    return Y, variance, rstd

def l2_norm_bwd(X, DY, variance, rstd, BLOCK_SIZE=256):
    DX = torch.zeros_like(X)
    
    grid = (X.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE, (X.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    _l2_norm_bwd_kernel[X.shape[0], BLOCK_SIZE](X, DY, DX, variance, rstd, BLOCK_SIZE=BLOCK_SIZE)
    
    return DX
