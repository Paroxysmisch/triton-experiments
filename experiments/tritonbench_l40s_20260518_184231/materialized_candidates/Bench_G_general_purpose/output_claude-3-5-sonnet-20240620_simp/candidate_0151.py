import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_fwd_1pass_kernel(
    # Pointers to tensors
    X, Y, W, B, MEAN, RSTD, RESIDUAL, X1, W1, B1, Y1, RESIDUAL_OUT, 
    DROPOUT_MASK, DROPOUT_MASK1, SEEDS,
    # Dimensions
    stride_xb, stride_xh,  # Strides for batch and hidden dimensions
    N,  # Hidden dimension size
    eps,  # Epsilon for numerical stability
    # Options
    IS_RMS_NORM: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_X1: tl.constexpr,
    STORE_RESIDUAL: tl.constexpr,
    DROPOUT_P: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute batch index and offsets
    batch_idx = pid
    offset_x = batch_idx * stride_xb + tl.arange(0, N) * stride_xh
    
    # Load input X
    x = tl.load(X + offset_x)
    
    # Compute mean and variance
    if not IS_RMS_NORM:
        # Regular layer norm: compute mean
        mean = tl.sum(x, axis=0) / N
        xm = x - mean
        # Compute variance
        x2 = xm * xm
    else:
        # RMS norm: no mean subtraction
        mean = 0
        x2 = x * x
        
    var = tl.sum(x2, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    
    # Load weight and bias if provided
    w = tl.load(W + tl.arange(0, N)) if W is not None else 1
    b = tl.load(B + tl.arange(0, N)) if B is not None else 0
    
    # Normalize and apply weight/bias
    if not IS_RMS_NORM:
        y = (x - mean) * rstd * w + b
    else:
        y = x * rstd * w + b
    
    # Handle residual connection
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + offset_x)
        if DROPOUT_P > 0.0:
            # Generate dropout mask for residual
            seed = tl.load(SEEDS + batch_idx)
            dropout_mask = tl.rand(seed) > DROPOUT_P
            residual = residual * dropout_mask * (1.0 / (1.0 - DROPOUT_P))
            tl.store(DROPOUT_MASK + offset_x, dropout_mask)
        y = y + residual
    
    # Handle X1 path if enabled
    if HAS_X1:
        x1 = tl.load(X1 + offset_x)
        w1 = tl.load(W1 + tl.arange(0, N)) if W1 is not None else 1
        b1 = tl.load(B1 + tl.arange(0, N)) if B1 is not None else 0
        
        y1 = x1 * w1 + b1
        if DROPOUT_P > 0.0:
            seed = tl.load(SEEDS + batch_idx + 1)  # Different seed for X1
            dropout_mask1 = tl.rand(seed) > DROPOUT_P
            y1 = y1 * dropout_mask1 * (1.0 / (1.0 - DROPOUT_P))
            tl.store(DROPOUT_MASK1 + offset_x, dropout_mask1)
        tl.store(Y1 + offset_x, y1)
    
    # Store outputs
    tl.store(Y + offset_x, y)
    if not IS_RMS_NORM:
        tl.store(MEAN + batch_idx, mean)
    tl.store(RSTD + batch_idx, rstd)
    
    # Store residual if needed
    if STORE_RESIDUAL:
        tl.store(RESIDUAL_OUT + offset_x, y)

# Wrapper function
def layer_norm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    residual: torch.Tensor = None,
    x1: torch.Tensor = None,
    weight1: torch.Tensor = None,
    bias1: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_rms_norm: bool = False,
    store_residual: bool = False,
):
    batch_size, hidden_size = x.shape
    device = x.device
    
    # Allocate output tensors
    y = torch.empty_like(x)
    mean = torch.empty(batch_size, device=device) if not is_rms_norm else None
    rstd = torch.empty(batch_size, device=device)
    
    # Optional outputs
    y1 = torch.empty_like(x) if x1 is not None else None
    residual_out = torch.empty_like(x) if store_residual else None
    dropout_mask = torch.empty_like(x, dtype=torch.bool) if dropout_p > 0 else None
    dropout_mask1 = torch.empty_like(x, dtype=torch.bool) if dropout_p > 0 and x1 is not None else None
    seeds = torch.randint(2**32, (batch_size * 2,), device=device) if dropout_p > 0 else None
    
    # Launch kernel
    grid = (batch_size,)
    _layer_norm_fwd_1pass_kernel[grid](
        x, y, weight, bias, mean, rstd, residual, 
        x1, weight1, bias1, y1, residual_out,
        dropout_mask, dropout_mask1, seeds,
        x.stride(0), x.stride(1),
        hidden_size, eps,
        is_rms_norm,
        residual is not None,
        x1 is not None,
        store_residual,
        dropout_p,
    )
    
    return (y, y1, mean, rstd, residual_out, seeds, dropout_mask, dropout_mask1)
