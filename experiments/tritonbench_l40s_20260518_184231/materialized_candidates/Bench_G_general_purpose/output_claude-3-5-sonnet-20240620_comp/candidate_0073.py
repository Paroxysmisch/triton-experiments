import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'NUM_WARPS': 4}),
        triton.Config({'BLOCK_SIZE_N': 256, 'NUM_WARPS': 8}),
        triton.Config({'BLOCK_SIZE_N': 512, 'NUM_WARPS': 16}),
    ],
    key=['N'],
)
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    Y,  # pointer to the output
    R,  # pointer to the residual (optional)
    Res,  # pointer to store residual output (optional)
    stride,  # stride between rows
    N,  # number of columns
    eps,  # epsilon added for numerical stability
    USE_RESIDUAL: tl.constexpr,  # whether to use residual connection
    STORE_RESIDUAL: tl.constexpr,  # whether to store residual output
    RMS_NORM: tl.constexpr,  # whether to use RMS normalization
    BLOCK_SIZE_N: tl.constexpr,  # number of columns to process in parallel
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    # Load data
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0)
    
    # Apply residual if needed
    if USE_RESIDUAL:
        r = tl.load(R + row * stride + cols, mask=mask, other=0.0)
        x += r

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N
    if RMS_NORM:
        var = tl.sum(x * x, axis=0) / N
    else:
        var = tl.sum((x - mean) ** 2, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Normalize
    y = (x - mean) * rstd if not RMS_NORM else x * rstd

    # Apply weight and bias
    w = tl.load(W + cols, mask=mask, other=1.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    y = y * w + b

    # Store output
    tl.store(Y + row * stride + cols, y, mask=mask)

    # Store residual output if needed
    if STORE_RESIDUAL:
        tl.store(Res + row * stride + cols, x, mask=mask)

@triton.jit
def _layer_norm_bwd_kernel(
    DY,  # pointer to the output gradient
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    DX,  # pointer to the input gradient
    DW,  # pointer to the weight gradient
    DB,  # pointer to the bias gradient
    Y,  # pointer to the forward output (for recomputation)
    R,  # pointer to the residual (optional)
    DR,  # pointer to the residual gradient (optional)
    stride,  # stride between rows
    N,  # number of columns
    eps,  # epsilon added for numerical stability
    USE_RESIDUAL: tl.constexpr,  # whether to use residual connection
    STORE_RESIDUAL_GRAD: tl.constexpr,  # whether to store residual gradient
    RMS_NORM: tl.constexpr,  # whether to use RMS normalization
    BLOCK_SIZE_N: tl.constexpr,  # number of columns to process in parallel
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    # Load data
    dy = tl.load(DY + row * stride + cols, mask=mask, other=0.0)
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=1.0)

    # Apply residual if needed
    if USE_RESIDUAL:
        r = tl.load(R + row * stride + cols, mask=mask, other=0.0)
        x += r

    # Recompute forward pass
    mean = tl.sum(x, axis=0) / N
    if RMS_NORM:
        var = tl.sum(x * x, axis=0) / N
    else:
        var = tl.sum((x - mean) ** 2, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    y = (x - mean) * rstd if not RMS_NORM else x * rstd

    # Compute gradients
    dw = tl.sum(dy * y, axis=0)
    db = tl.sum(dy, axis=0)
    dy = dy * w

    if RMS_NORM:
        dx = dy * rstd - (y * tl.sum(dy * y, axis=0)) / N
    else:
        dx = dy * rstd - (tl.sum(dy, axis=0) + y * tl.sum(dy * y, axis=0)) / N

    # Store gradients
    tl.store(DX + row * stride + cols, dx, mask=mask)
    tl.atomic_add(DW + cols, dw, mask=mask)
    tl.atomic_add(DB + cols, db, mask=mask)

    # Store residual gradient if needed
    if STORE_RESIDUAL_GRAD:
        tl.store(DR + row * stride + cols, dx, mask=mask)

# Wrapper function for forward pass
def layer_norm_forward(x, weight, bias, eps=1e-5, use_residual=False, store_residual=False, rms_norm=False):
    M, N = x.shape
    y = torch.empty_like(x)
    residual_out = torch.empty_like(x) if store_residual else None
    
    # Determine BLOCK_SIZE_N based on N and memory constraints
    BLOCK_SIZE_N = min(triton.next_power_of_2(N), 1024)
    
    # Launch kernel
    grid = (M,)
    _layer_norm_fwd_1pass_kernel[grid](
        x, weight, bias, y,
        x if use_residual else None,  # Use input as residual if needed
        residual_out,
        x.stride(0),
        N,
        eps,
        USE_RESIDUAL=use_residual,
        STORE_RESIDUAL=store_residual,
        RMS_NORM=rms_norm,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return y, residual_out

# Wrapper function for backward pass
def layer_norm_backward(dy, x, weight, bias, eps=1e-5, use_residual=False, store_residual_grad=False, rms_norm=False):
    M, N = x.shape
    dx = torch.empty_like(x)
    dw = torch.zeros_like(weight)
    db = torch.zeros_like(bias)
    dr = torch.empty_like(x) if store_residual_grad else None
    
    # Determine BLOCK_SIZE_N based on N and memory constraints
    BLOCK_SIZE_N = min(triton.next_power_of_2(N), 1024)
    
    # Launch kernel
    grid = (M,)
    _layer_norm_bwd_kernel[grid](
        dy, x, weight, bias, dx, dw, db,
        None,  # Y is recomputed in the kernel
        x if use_residual else None,  # Use input as residual if needed
        dr,
        x.stride(0),
        N,
        eps,
        USE_RESIDUAL=use_residual,
        STORE_RESIDUAL_GRAD=store_residual_grad,
        RMS_NORM=rms_norm,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return dx, dw, db, dr
