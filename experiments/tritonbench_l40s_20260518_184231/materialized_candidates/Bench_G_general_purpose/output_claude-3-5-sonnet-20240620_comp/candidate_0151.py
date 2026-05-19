import triton
import triton.language as tl
import torch

@triton.jit
def _layer_norm_fwd_1pass_kernel(
    # Pointers to matrices
    X, Y, W, B,                    # Main inputs/outputs
    MEAN, INV_STD,                 # Statistics
    RESIDUAL, X1, W1, B1, Y1,      # Optional inputs/outputs
    ROWSCALE, SEEDS, DROPOUT_MASK, # Additional parameters
    # Matrix dimensions
    M, N,                          
    # Parameters
    eps,                           # Epsilon for numerical stability
    dropout_p,                     # Dropout probability
    rms_norm,                      # RMS normalization flag
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Map program id to the row of the matrix
    row = tl.program_id(0)
    
    # Compute row offset
    row_start_ptr = row * N
    
    # Load data for the current row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Load input X
    x = tl.load(X + row_start_ptr + cols, mask=mask, other=0.0)
    
    # Step 1: Compute mean (skip for RMS norm)
    if not rms_norm:
        mean = tl.sum(x, axis=0) / N
    else:
        mean = 0.0
    
    # Step 2: Compute variance
    x_centered = x - mean if not rms_norm else x
    var = tl.sum(x_centered * x_centered, axis=0) / N
    
    # Step 3: Compute inverse standard deviation
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Step 4: Normalize
    x_norm = x_centered * inv_std
    
    # Step 5: Apply scale and bias
    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    y = x_norm * w + b
    
    # Apply dropout if needed
    if dropout_p > 0.0:
        seed = tl.load(SEEDS + row)
        rand = tl.rand(seed, cols)
        dropout_mask = rand > dropout_p
        y = tl.where(dropout_mask, y / (1.0 - dropout_p), 0.0)
        # Store dropout mask if needed
        if DROPOUT_MASK is not None:
            tl.store(DROPOUT_MASK + row_start_ptr + cols, dropout_mask, mask=mask)
    
    # Handle residual connection
    if RESIDUAL is not None:
        residual = tl.load(RESIDUAL + row_start_ptr + cols, mask=mask, other=0.0)
        y = y + residual
    
    # Apply row scaling if needed
    if ROWSCALE is not None:
        scale = tl.load(ROWSCALE + row)
        y = y * scale
    
    # Store results
    tl.store(Y + row_start_ptr + cols, y, mask=mask)
    
    # Store statistics if needed
    if MEAN is not None and not rms_norm:
        tl.store(MEAN + row, mean)
    if INV_STD is not None:
        tl.store(INV_STD + row, inv_std)
    
    # Handle optional second output (Y1) if needed
    if Y1 is not None and W1 is not None and B1 is not None:
        w1 = tl.load(W1 + cols, mask=mask, other=0.0)
        b1 = tl.load(B1 + cols, mask=mask, other=0.0)
        y1 = x_norm * w1 + b1
        tl.store(Y1 + row_start_ptr + cols, y1, mask=mask)

# Wrapper function
def layer_norm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    dropout_p: float = 0.0,
    residual: torch.Tensor = None,
    rowscale: torch.Tensor = None,
    rms_norm: bool = False,
) -> tuple:
    # Get dimensions
    M, N = x.shape
    
    # Allocate output tensors
    y = torch.empty_like(x)
    mean = torch.empty(M, device=x.device) if not rms_norm else None
    inv_std = torch.empty(M, device=x.device)
    
    # Prepare dropout if needed
    if dropout_p > 0.0:
        seeds = torch.randint(0, 2**31 - 1, (M,), device=x.device)
        dropout_mask = torch.empty_like(x, dtype=torch.bool)
    else:
        seeds = None
        dropout_mask = None
    
    # Determine grid and block sizes
    BLOCK_SIZE = min(triton.next_power_of_2(N), 1024)
    grid = (M,)
    
    # Launch kernel
    _layer_norm_fwd_1pass_kernel[grid](
        x, y, weight, bias,                      # Main inputs/outputs
        mean, inv_std,                           # Statistics
        residual, None, None, None, None,        # Optional inputs (simplified)
        rowscale, seeds, dropout_mask,           # Additional parameters
        M, N,                                    # Dimensions
        eps, dropout_p, rms_norm,                # Parameters
        BLOCK_SIZE=BLOCK_SIZE,                   # Meta-parameters
    )
    
    return y, mean, inv_std, dropout_mask
