import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_forward_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr, Mean_ptr, Rstd_ptr,
    stride_x_row, stride_x_col,
    stride_w, stride_b,
    stride_y_row, stride_y_col,
    N_ROWS, N_COLS,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute row pointer
    row_start_ptr = X_ptr + pid * stride_x_row
    
    # Initialize accumulators for mean and m2
    mean = 0.0
    m2 = 0.0
    
    # Load data and compute mean/variance
    for idx in range(0, N_COLS, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE) + idx
        mask = cols < N_COLS
        x = tl.load(row_start_ptr + cols * stride_x_col, mask=mask, other=0.0)
        mean += tl.sum(x, axis=0)
        m2 += tl.sum(x * x, axis=0)
    
    # Finalize statistics
    mean = mean / N_COLS
    variance = m2/N_COLS - mean*mean
    rstd = 1 / tl.sqrt(variance + eps)
    
    # Store mean and rstd
    tl.store(Mean_ptr + pid, mean)
    tl.store(Rstd_ptr + pid, rstd)
    
    # Normalize and apply weight/bias
    row_y_ptr = Y_ptr + pid * stride_y_row
    for idx in range(0, N_COLS, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE) + idx
        mask = cols < N_COLS
        x = tl.load(row_start_ptr + cols * stride_x_col, mask=mask)
        w = tl.load(W_ptr + cols * stride_w, mask=mask)
        b = tl.load(B_ptr + cols * stride_b, mask=mask)
        
        y = (x - mean) * rstd
        y = y * w + b
        tl.store(row_y_ptr + cols * stride_y_col, y, mask=mask)

def layer_norm_forward(x, weight, bias, eps=1e-5):
    # Get input dimensions
    batch_size, hidden_size = x.shape
    
    # Create output tensors
    y = torch.empty_like(x)
    mean = torch.empty(batch_size, device=x.device, dtype=x.dtype)
    rstd = torch.empty(batch_size, device=x.device, dtype=x.dtype)
    
    # Calculate kernel settings
    BLOCK_SIZE = min(hidden_size, 1024)
    num_warps = 4
    
    # Launch kernel
    grid = (batch_size,)
    _layer_norm_forward_kernel[grid](
        x, weight, bias, y, mean, rstd,
        x.stride(0), x.stride(1),
        weight.stride(0), bias.stride(0),
        y.stride(0), y.stride(1),
        batch_size, hidden_size,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return y, mean, rstd

@triton.jit
def _layer_norm_backward_kernel(
    DY_ptr, X_ptr, W_ptr, Mean_ptr, Rstd_ptr,
    DX_ptr, DW_ptr, DB_ptr,
    stride_dy_row, stride_dy_col,
    stride_x_row, stride_x_col,
    stride_w,
    stride_dx_row, stride_dx_col,
    N_ROWS, N_COLS,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Load statistics
    mean = tl.load(Mean_ptr + pid)
    rstd = tl.load(Rstd_ptr + pid)
    
    # Initialize accumulators
    sum_dy = 0.0
    sum_dy_x_minus_mean = 0.0
    
    # Load row pointers
    row_dy_ptr = DY_ptr + pid * stride_dy_row
    row_x_ptr = X_ptr + pid * stride_x_row
    
    # First pass: compute reduction terms
    for idx in range(0, N_COLS, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE) + idx
        mask = cols < N_COLS
        
        dy = tl.load(row_dy_ptr + cols * stride_dy_col, mask=mask, other=0.0)
        x = tl.load(row_x_ptr + cols * stride_x_col, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols * stride_w, mask=mask, other=0.0)
        
        x_centered = x - mean
        sum_dy += tl.sum(dy * w, axis=0)
        sum_dy_x_minus_mean += tl.sum(dy * w * x_centered, axis=0)
    
    # Compute gradient factors
    factor1 = rstd
    factor2 = sum_dy_x_minus_mean * rstd * rstd * rstd / N_COLS
    factor3 = sum_dy / N_COLS
    
    # Second pass: compute gradients
    row_dx_ptr = DX_ptr + pid * stride_dx_row
    for idx in range(0, N_COLS, BLOCK_SIZE):
        cols = tl.arange(0, BLOCK_SIZE) + idx
        mask = cols < N_COLS
        
        dy = tl.load(row_dy_ptr + cols * stride_dy_col, mask=mask)
        x = tl.load(row_x_ptr + cols * stride_x_col, mask=mask)
        w = tl.load(W_ptr + cols * stride_w, mask=mask)
        
        x_centered = x - mean
        dx = w * (dy * factor1 - x_centered * factor2 - factor3)
        
        # Store dx
        tl.store(row_dx_ptr + cols * stride_dx_col, dx, mask=mask)
        
        # Accumulate dw and db
        if pid == 0:
            dw = dy * (x - mean) * rstd
            db = dy
            tl.atomic_add(DW_ptr + cols * stride_w, dw, mask=mask)
            tl.atomic_add(DB_ptr + cols * stride_w, db, mask=mask)

def layer_norm_backward(grad_output, x, weight, mean, rstd):
    batch_size, hidden_size = x.shape
    
    # Create output tensors
    dx = torch.empty_like(x)
    dw = torch.zeros_like(weight)
    db = torch.zeros_like(weight)
    
    # Calculate kernel settings
    BLOCK_SIZE = min(hidden_size, 1024)
    num_warps = 4
    
    # Launch kernel
    grid = (batch_size,)
    _layer_norm_backward_kernel[grid](
        grad_output, x, weight, mean, rstd,
        dx, dw, db,
        grad_output.stride(0), grad_output.stride(1),
        x.stride(0), x.stride(1),
        weight.stride(0),
        dx.stride(0), dx.stride(1),
        batch_size, hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return dx, dw, db

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        y, mean, rstd = layer_norm_forward(x, weight, bias, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, mean, rstd = ctx.saved_tensors
        dx, dw, db = layer_norm_backward(grad_output, x, weight, mean, rstd)
        return dx, dw, db, None

class TritonLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
