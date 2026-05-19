import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.jit
def _layer_norm_fwd_quant_kernel(
    # ... existing input parameters ...
):
    # Map program ID to row
    row = tl.program_id(0)
    
    # Load input data
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    
    # Handle residual connection if present
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
        
    # Compute statistics
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
        
    rstd = 1 / tl.sqrt(var + eps)
    
    # Normalize and transform
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w if HAS_WEIGHT else x_hat
    if HAS_BIAS:
        y = y + b
        
    # Quantization
    scale = 127.0 / tl.maximum(tl.max(tl.abs(y), 0), 1e-5)
    y = tl.math.round(y * scale)
    y = tl.maximum(tl.minimum(y, 127), -128) / scale
    
    # Store results
    tl.store(Y + cols, y, mask=cols < N)

@triton.jit
def _layer_norm_bwd_kernel(
    # ... existing parameters ...
):
    row_block_id = tl.program_id(0)
    
    # Load data
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    
    # Compute gradients
    if not IS_RMS_NORM:
        c1 = tl.sum(xhat * wdy, axis=0) / N
        c2 = tl.sum(wdy, axis=0) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd
    else:
        c1 = tl.sum(xhat * wdy, axis=0) / N
        dx = (wdy - xhat * c1) * rstd
        
    # Handle residual gradients
    if HAS_DRESIDUAL:
        dx += dres
