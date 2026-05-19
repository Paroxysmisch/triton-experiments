import torch
import triton
import triton.language as tl

# Forward pass kernel
@triton.jit
def _layer_norm_fwd_fused(X, W, B, Y, MEAN, RSTD, M, N, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Load data
    x_ptrs = X + row_idx * N + cols
    x = tl.load(x_ptrs, mask=cols < N, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    
    # Compute variance and std
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    
    # Normalize
    x_norm = x_centered * rstd
    
    # Apply scale and shift
    w_ptrs = W + cols
    b_ptrs = B + cols
    w = tl.load(w_ptrs, mask=cols < N)
    b = tl.load(b_ptrs, mask=cols < N)
    
    y = x_norm * w + b
    y_ptrs = Y + row_idx * N + cols
    tl.store(y_ptrs, y, mask=cols < N)
    
    # Store mean and rstd for backward pass
    tl.store(MEAN + row_idx, mean)
    tl.store(RSTD + row_idx, rstd)

# Backward pass kernel for input gradient
@triton.jit
def _layer_norm_bwd_dx_fused(DY, X, MEAN, RSTD, W, DX, DW, DB, M, N, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Load data
    dy_ptrs = DY + row_idx * N + cols
    dy = tl.load(dy_ptrs, mask=cols < N, other=0.0)
    
    x_ptrs = X + row_idx * N + cols
    x = tl.load(x_ptrs, mask=cols < N, other=0.0)
    
    mean = tl.load(MEAN + row_idx)
    rstd = tl.load(RSTD + row_idx)
    
    w_ptrs = W + cols
    w = tl.load(w_ptrs, mask=cols < N)
    
    # Compute dx
    x_centered = x - mean
    x_norm = x_centered * rstd
    dx_norm = dy * w
    dx_centered = dx_norm * rstd
    dx = dx_centered - tl.sum(dx_centered, axis=0) / N - x_norm * tl.sum(dx_centered * x_norm, axis=0) / N
    dx_ptrs = DX + row_idx * N + cols
    tl.store(dx_ptrs, dx, mask=cols < N)
    
    # Compute partial dw and db
    dw_partial = dy * x_norm
    db_partial = dy
    
    # Atomic add for dw and db
    tl.atomic_add(DW + cols, dw_partial, mask=cols < N)
    tl.atomic_add(DB + cols, db_partial, mask=cols < N)

# Backward pass kernel for weight and bias gradients
@triton.jit
def _layer_norm_bwd_dwdb(DW_PARTIAL, DB_PARTIAL, FINAL_DW, FINAL_DB, M, N, BLOCK_SIZE: tl.constexpr):
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Aggregate partial sums
    dw_partial = tl.load(DW_PARTIAL + cols, mask=cols < N, other=0.0)
    db_partial = tl.load(DB_PARTIAL + cols, mask=cols < N, other=0.0)
    
    # Store final dw and db
    tl.store(FINAL_DW + cols, dw_partial, mask=cols < N)
    tl.store(FINAL_DB + cols, db_partial, mask=cols < N)

# Wrapper class for LayerNorm
class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B):
        M, N = X.shape
        BLOCK_SIZE = 1024  # This can be tuned based on the hardware

        # Allocate output tensors
        Y = torch.empty_like(X)
        MEAN = torch.empty(M, device=X.device, dtype=X.dtype)
        RSTD = torch.empty(M, device=X.device, dtype=X.dtype)

        # Launch forward kernel
        _layer_norm_fwd_fused[(M,)](X, W, B, Y, MEAN, RSTD, M, N, BLOCK_SIZE=BLOCK_SIZE)

        # Save context for backward
        ctx.save_for_backward(X, W, MEAN, RSTD)
        return Y

    @staticmethod
    def backward(ctx, DY):
        X, W, MEAN, RSTD = ctx.saved_tensors
        M, N = X.shape
        BLOCK_SIZE = 1024

        # Allocate gradient tensors
        DX = torch.empty_like(X)
        DW_PARTIAL = torch.zeros_like(W)
        DB_PARTIAL = torch.zeros_like(B)

        # Launch backward kernel for input gradient
        _layer_norm_bwd_dx_fused[(M,)](DY, X, MEAN, RSTD, W, DX, DW_PARTIAL, DB_PARTIAL, M, N, BLOCK_SIZE=BLOCK_SIZE)

        # Allocate final gradients
        FINAL_DW = torch.empty_like(W)
        FINAL_DB = torch.empty_like(B)

        # Launch backward kernel for weight and bias gradients
        _layer_norm_bwd_dwdb[(1,)](DW_PARTIAL, DB_PARTIAL, FINAL_DW, FINAL_DB, M, N, BLOCK_SIZE=BLOCK_SIZE)

        return DX, FINAL_DW, FINAL_DB

# Usage example
def layer_norm(x, w, b):
    return LayerNorm.apply(x, w, b)
