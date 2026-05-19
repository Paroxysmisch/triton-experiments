import torch
import triton
import triton.language as tl
import math

# Constants
MAX_FUSED_SIZE = 65536  # Maximum size for fused operations

@triton.jit
def _layer_norm_fwd_fused(
    X_ptr, W_ptr, B_ptr, Y_ptr, Mean_ptr, Rstd_ptr,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute offsets
    offs = pid * stride + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Load data
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    w = tl.load(W_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    b = tl.load(B_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / N
    
    # Compute variance
    x_mean = x - mean
    x_var = tl.sum(x_mean * x_mean, axis=0) / N
    rstd = 1 / tl.sqrt(x_var + eps)
    
    # Normalize and apply scale/bias
    y = w * (x_mean * rstd) + b
    
    # Store results
    tl.store(Y_ptr + offs, y, mask=mask)
    tl.store(Mean_ptr + pid, mean)
    tl.store(Rstd_ptr + pid, rstd)

@triton.jit
def _layer_norm_bwd_fused(
    dY_ptr, X_ptr, W_ptr, Mean_ptr, Rstd_ptr, 
    dX_ptr, dW_ptr, dB_ptr,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * stride + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Load data
    dy = tl.load(dY_ptr + offs, mask=mask, other=0.0)
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    w = tl.load(W_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    mean = tl.load(Mean_ptr + pid)
    rstd = tl.load(Rstd_ptr + pid)
    
    # Compute gradients
    x_hat = (x - mean) * rstd
    dw = tl.sum(dy * x_hat, axis=0)
    db = tl.sum(dy, axis=0)
    
    dx_hat = dy * w
    dvar = -0.5 * rstd * rstd * rstd * tl.sum(dx_hat * (x - mean), axis=0)
    dmean = -rstd * tl.sum(dx_hat, axis=0)
    
    dx = dx_hat * rstd + (2 * dvar * (x - mean) + dmean) / N
    
    # Store results
    tl.store(dX_ptr + offs, dx, mask=mask)
    tl.store(dW_ptr + tl.arange(0, BLOCK_SIZE), dw, mask=mask)
    tl.store(dB_ptr + tl.arange(0, BLOCK_SIZE), db, mask=mask)

def calculate_settings(n):
    """Calculate optimal block size and number of warps"""
    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n))
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16
    return block_size, num_warps

class FastLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        # Calculate settings
        batch_size, n = x.shape
        block_size, num_warps = calculate_settings(n)
        
        # Allocate output
        y = torch.empty_like(x)
        mean = torch.empty(batch_size, dtype=x.dtype, device=x.device)
        rstd = torch.empty(batch_size, dtype=x.dtype, device=x.device)
        
        # Launch kernel
        grid = (batch_size,)
        _layer_norm_fwd_fused[grid](
            x, weight, bias, y, mean, rstd,
            n, n, eps,
            BLOCK_SIZE=block_size,
            num_warps=num_warps
        )
        
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, mean, rstd = ctx.saved_tensors
        eps = ctx.eps
        
        # Calculate settings
        batch_size, n = x.shape
        block_size, num_warps = calculate_settings(n)
        
        # Allocate gradients
        grad_x = torch.empty_like(x)
        grad_weight = torch.empty_like(weight)
        grad_bias = torch.empty_like(weight)
        
        # Launch kernel
        grid = (batch_size,)
        _layer_norm_bwd_fused[grid](
            grad_output, x, weight, mean, rstd,
            grad_x, grad_weight, grad_bias,
            n, n, eps,
            BLOCK_SIZE=block_size,
            num_warps=num_warps
        )
        
        return grad_x, grad_weight, grad_bias, None

def fast_layernorm(x: torch.Tensor, normalized_shape: int, 
                   weight: torch.Tensor, bias: torch.Tensor, 
                   eps: float = 1e-5) -> torch.Tensor:
    """
    Applies layer normalization using the custom Triton implementation
    """
    return FastLayerNorm.apply(x, weight, bias, eps)
