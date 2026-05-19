import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_fwd_1pass_kernel(
    # Pointers to matrices
    X,  # Input tensor (B, M)
    Y,  # Output tensor (B, M)
    W,  # Weight tensor (M,)
    B,  # Bias tensor (M,)
    Mean,  # Mean tensor (B, 1)
    Rstd,  # 1/std tensor (B, 1)
    # Matrix dimensions
    stride_xb, stride_xm,  # Strides for input X
    stride_yb, stride_ym,  # Strides for output Y
    # Parameters
    eps,  # Epsilon for numerical stability
    rms_norm,  # Whether to use RMS normalization
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Offset calculations
    offset_x = pid * stride_xb + tl.arange(0, BLOCK_SIZE) * stride_xm
    
    # Load input
    mask = tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE
    x = tl.load(X + offset_x, mask=mask)
    
    # Calculate mean
    if not rms_norm:
        mean = tl.sum(x, axis=0) / BLOCK_SIZE
        tl.store(Mean + pid, mean)
    else:
        mean = 0.0
        
    # Calculate variance
    x_centered = x - mean
    x_var = tl.sum(x_centered * x_centered, axis=0) / BLOCK_SIZE
    rstd = 1 / tl.sqrt(x_var + eps)
    tl.store(Rstd + pid, rstd)
    
    # Normalize
    x_norm = x_centered * rstd
    
    # Load weights and bias
    w = tl.load(W + tl.arange(0, BLOCK_SIZE))
    b = tl.load(B + tl.arange(0, BLOCK_SIZE))
    
    # Apply weight and bias
    y = x_norm * w + b
    
    # Store output
    tl.store(Y + offset_x, y, mask=mask)

@triton.jit
def _layer_norm_bwd_kernel(
    # Pointers to matrices
    DY,  # Gradient of output (B, M)
    X,   # Input tensor (B, M)
    W,   # Weight tensor (M,)
    Mean,  # Mean tensor (B, 1)
    Rstd,  # 1/std tensor (B, 1)
    DX,   # Gradient of input (B, M)
    DW,   # Gradient of weights (M,)
    DB,   # Gradient of bias (M,)
    # Matrix dimensions
    stride_dyb, stride_dym,  # Strides for DY
    stride_xb, stride_xm,    # Strides for X
    stride_dxb, stride_dxm,  # Strides for DX
    # Parameters
    eps,  # Epsilon for numerical stability
    rms_norm,  # Whether to use RMS normalization
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Load input gradients
    offset_dy = pid * stride_dyb + tl.arange(0, BLOCK_SIZE) * stride_dym
    mask = tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE
    dy = tl.load(DY + offset_dy, mask=mask)
    
    # Load saved forward pass values
    x = tl.load(X + pid * stride_xb + tl.arange(0, BLOCK_SIZE) * stride_xm, mask=mask)
    mean = tl.load(Mean + pid) if not rms_norm else 0.0
    rstd = tl.load(Rstd + pid)
    w = tl.load(W + tl.arange(0, BLOCK_SIZE))
    
    # Compute gradients
    x_centered = x - mean
    x_norm = x_centered * rstd
    
    # Gradient w.r.t. normalized input
    dx_norm = dy * w
    
    # Gradient w.r.t. variance
    dvar = -0.5 * tl.sum(dx_norm * x_norm * rstd, axis=0)
    
    # Gradient w.r.t. mean
    if not rms_norm:
        dmean = -tl.sum(dx_norm * rstd, axis=0)
        dmean = dmean - 2.0 * dvar * tl.sum(x_centered, axis=0) / BLOCK_SIZE
    else:
        dmean = 0.0
    
    # Gradient w.r.t. input
    dx = dx_norm * rstd
    if not rms_norm:
        dx = dx + (dmean + 2.0 * dvar * x_centered) / BLOCK_SIZE
    
    # Store input gradients
    tl.store(DX + pid * stride_dxb + tl.arange(0, BLOCK_SIZE) * stride_dxm, dx, mask=mask)
    
    # Accumulate weight and bias gradients
    dw = tl.sum(dy * x_norm, axis=0)
    db = tl.sum(dy, axis=0)
    
    # Store weight and bias gradients (first thread only)
    if pid == 0:
        tl.store(DW + tl.arange(0, BLOCK_SIZE), dw)
        tl.store(DB + tl.arange(0, BLOCK_SIZE), db)

# Wrapper function for the forward pass
def layer_norm_forward(x, weight, bias, eps=1e-5, rms_norm=False):
    batch, hidden = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((batch, 1), device=x.device)
    rstd = torch.empty((batch, 1), device=x.device)
    
    # Configure grid and block sizes
    grid = (batch,)
    _layer_norm_fwd_1pass_kernel[grid](
        x, y, weight, bias, mean, rstd,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        eps, rms_norm,
        BLOCK_SIZE=hidden,
    )
    
    return y, mean, rstd

# Wrapper function for the backward pass
def layer_norm_backward(dy, x, weight, mean, rstd, eps=1e-5, rms_norm=False):
    batch, hidden = x.shape
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    db = torch.empty_like(bias)
    
    # Configure grid and block sizes
    grid = (batch,)
    _layer_norm_bwd_kernel[grid](
        dy, x, weight, mean, rstd, dx, dw, db,
        dy.stride(0), dy.stride(1),
        x.stride(0), x.stride(1),
        dx.stride(0), dx.stride(1),
        eps, rms_norm,
        BLOCK_SIZE=hidden,
    )
    
    return dx, dw, db
