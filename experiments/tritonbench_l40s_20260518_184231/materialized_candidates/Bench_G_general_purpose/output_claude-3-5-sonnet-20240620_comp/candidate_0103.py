import torch
import triton
import triton.language as tl
import math

@triton.jit
def _rms_layernorm_forward(
    X_ptr, W_ptr, Y_ptr, Var_ptr,
    stride_xb, stride_xm, stride_wb,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute offsets
    offset_x = pid * stride_xb
    offset_w = pid * stride_wb
    
    # Load X and compute variance
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    x = tl.load(X_ptr + offset_x + cols * stride_xm, mask=mask, other=0.0)
    x2 = x * x
    
    # Compute variance
    var = tl.sum(x2, axis=0) / N
    var = var + eps
    
    # Store variance for backward pass
    tl.store(Var_ptr + pid, var)
    
    # Normalize and scale
    inv_var = 1.0 / tl.sqrt(var)
    w = tl.load(W_ptr + offset_w + cols, mask=mask, other=1.0)
    
    y = x * inv_var * w
    
    # Store output
    tl.store(Y_ptr + offset_x + cols * stride_xm, y, mask=mask)

@triton.jit
def _rms_layernorm_backward(
    DY_ptr, X_ptr, W_ptr, Var_ptr,
    DX_ptr, DW_ptr,
    stride_xb, stride_xm, stride_wb,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Load data
    offset_x = pid * stride_xb
    offset_w = pid * stride_wb
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    dy = tl.load(DY_ptr + offset_x + cols * stride_xm, mask=mask, other=0.0)
    x = tl.load(X_ptr + offset_x + cols * stride_xm, mask=mask, other=0.0)
    w = tl.load(W_ptr + offset_w + cols, mask=mask, other=1.0)
    var = tl.load(Var_ptr + pid)
    
    # Compute gradients
    inv_var = 1.0 / tl.sqrt(var)
    x_hat = x * inv_var
    
    # Gradient w.r.t weights
    dw = x_hat * dy
    tl.store(DW_ptr + offset_w + cols, dw, mask=mask)
    
    # Gradient w.r.t input
    dx = dy * w * inv_var
    dx = dx - (x * tl.sum(dx * x, axis=0)) / (N * var)
    
    tl.store(DX_ptr + offset_x + cols * stride_xm, dx, mask=mask)

@triton.jit
def _gemma_rms_layernorm_forward(
    X_ptr, W_ptr, Y_ptr, Var_ptr,
    stride_xb, stride_xm, stride_wb,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offset_x = pid * stride_xb
    offset_w = pid * stride_wb
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    x = tl.load(X_ptr + offset_x + cols * stride_xm, mask=mask, other=0.0)
    x2 = x * x
    var = tl.sum(x2, axis=0) / N
    var = var + eps
    
    tl.store(Var_ptr + pid, var)
    
    inv_var = 1.0 / tl.sqrt(var)
    w = tl.load(W_ptr + offset_w + cols, mask=mask, other=1.0)
    
    # Add 1.0 to weight for Gemma-style RMS LayerNorm
    y = x * inv_var * (w + 1.0)
    
    tl.store(Y_ptr + offset_x + cols * stride_xm, y, mask=mask)

def calculate_settings(N):
    # Calculate optimal block size and number of warps
    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 4096:
        num_warps = 16
    return BLOCK_SIZE, num_warps

class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6, gemma_style=False):
        # Save input for backward pass
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        
        # Get dimensions
        batch_size, hidden_size = x.shape
        BLOCK_SIZE, num_warps = calculate_settings(hidden_size)
        
        # Allocate output
        y = torch.empty_like(x)
        var = torch.empty(batch_size, device=x.device, dtype=x.dtype)
        
        # Launch kernel
        grid = (batch_size,)
        kernel = _gemma_rms_layernorm_forward if gemma_style else _rms_layernorm_forward
        kernel[grid](
            x, weight, y, var,
            x.stride(0), x.stride(1), weight.stride(0),
            hidden_size, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        
        ctx.var = var
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        var = ctx.var
        eps = ctx.eps
        
        batch_size, hidden_size = x.shape
        BLOCK_SIZE, num_warps = calculate_settings(hidden_size)
        
        # Allocate gradients
        grad_x = torch.empty_like(x)
        grad_weight = torch.empty_like(weight)
        
        grid = (batch_size,)
        _rms_layernorm_backward[grid](
            grad_output, x, weight, var,
            grad_x, grad_weight,
            x.stride(0), x.stride(1), weight.stride(0),
            hidden_size, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        
        return grad_x, grad_weight, None, None

def fast_rms_layernorm(x, weight, eps=1e-6, gemma_style=False):
    return Fast_RMS_Layernorm.apply(x, weight, eps, gemma_style)
