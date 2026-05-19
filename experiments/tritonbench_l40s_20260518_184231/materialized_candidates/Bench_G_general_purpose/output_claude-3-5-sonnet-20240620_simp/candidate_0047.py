import torch
import triton
import triton.language as tl
import math

# Utility function to calculate optimal execution settings
def calculate_settings(BLOCK_SIZE):
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    return num_warps

# Forward kernel
@triton.jit
def layernorm_forward_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr, Mean_ptr, Rstd_ptr,
    stride, N_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    row_idx = tl.program_id(0)
    
    # Compute memory offsets
    row_start_ptr = X_ptr + row_idx * stride
    
    # Create load mask for the current row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_cols
    
    # Load data
    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + col_offsets, mask=mask, other=0.0)
    
    # Calculate mean
    mean = tl.sum(x, axis=0) / N_cols
    
    # Calculate variance
    x_mean = x - mean
    x_var = tl.sum(x_mean * x_mean, axis=0) / N_cols
    rstd = 1 / tl.sqrt(x_var + eps)
    
    # Normalize and apply scale and bias
    y = w * (x_mean * rstd) + b
    
    # Store results
    y_ptr = Y_ptr + row_idx * stride
    tl.store(y_ptr + col_offsets, y, mask=mask)
    
    # Store mean and rstd for backward pass
    tl.store(Mean_ptr + row_idx, mean)
    tl.store(Rstd_ptr + row_idx, rstd)

# Backward kernel
@triton.jit
def layernorm_backward_kernel(
    DY_ptr, X_ptr, W_ptr, Mean_ptr, Rstd_ptr,
    DX_ptr, DW_ptr, DB_ptr,
    stride, N_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    # Compute memory offsets
    row_start_ptr = DY_ptr + row_idx * stride
    x_row_start_ptr = X_ptr + row_idx * stride
    
    # Create load mask
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_cols
    
    # Load data
    dy = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
    x = tl.load(x_row_start_ptr + col_offsets, mask=mask, other=0.0)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    mean = tl.load(Mean_ptr + row_idx)
    rstd = tl.load(Rstd_ptr + row_idx)
    
    # Compute x_hat
    x_hat = (x - mean) * rstd
    
    # Compute gradients
    dy_w = dy * w
    dx_hat = dy_w * rstd
    dw = tl.sum(dy * x_hat, axis=0)
    db = tl.sum(dy, axis=0)
    
    # Compute dx
    dx = dx_hat - tl.sum(dx_hat, axis=0) / N_cols - x_hat * tl.sum(dx_hat * x_hat, axis=0) / N_cols
    
    # Store results
    dx_ptr = DX_ptr + row_idx * stride
    tl.store(dx_ptr + col_offsets, dx, mask=mask)
    tl.atomic_add(DW_ptr + col_offsets, dw, mask=mask)
    tl.atomic_add(DB_ptr + col_offsets, db, mask=mask)

# PyTorch autograd function
class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        # Save input for backward pass
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        
        # Get dimensions
        batch_size, n_cols = x.shape
        
        # Allocate output
        y = torch.empty_like(x)
        mean = torch.empty(batch_size, device=x.device)
        rstd = torch.empty(batch_size, device=x.device)
        
        # Calculate optimal block size and warps
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calculate_settings(BLOCK_SIZE)
        
        # Launch kernel
        grid = (batch_size,)
        layernorm_forward_kernel[grid](
            x, weight, bias, y, mean, rstd,
            x.stride(0), n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        
        ctx.mean = mean
        ctx.rstd = rstd
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        mean = ctx.mean
        rstd = ctx.rstd
        
        # Get dimensions
        batch_size, n_cols = grad_output.shape
        
        # Allocate gradients
        grad_x = torch.empty_like(x)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(weight)
        
        # Calculate optimal block size and warps
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = calculate_settings(BLOCK_SIZE)
        
        # Launch kernel
        grid = (batch_size,)
        layernorm_backward_kernel[grid](
            grad_output, x, weight, mean, rstd,
            grad_x, grad_weight, grad_bias,
            x.stride(0), n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        
        return grad_x, grad_weight, grad_bias, None

# Utility function
def fast_layernorm(x, layernorm_module):
    return Fast_Layernorm.apply(
        x,
        layernorm_module.weight,
        layernorm_module.bias,
        layernorm_module.eps
    )
