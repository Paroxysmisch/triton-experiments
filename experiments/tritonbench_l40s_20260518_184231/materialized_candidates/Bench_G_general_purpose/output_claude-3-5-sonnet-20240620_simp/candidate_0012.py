import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_forward_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr, Mean_ptr, Rstd_ptr,
    stride, n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    row_idx = tl.program_id(0)
    
    # Compute memory offsets for this row
    row_start_ptr = X_ptr + row_idx * stride
    
    # Initialize accumulators for mean and m2
    mean = 0.0
    m2 = 0.0
    
    # Load data and compute mean
    for col in range(0, n_cols, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(row_start_ptr + cols, mask=mask, other=0.0)
        mean += tl.sum(x, mask=mask)
    
    mean = mean / n_cols
    
    # Compute variance
    for col in range(0, n_cols, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(row_start_ptr + cols, mask=mask, other=0.0)
        diff = x - mean
        m2 += tl.sum(diff * diff, mask=mask)
    
    rstd = 1.0 / tl.sqrt(m2 / n_cols + eps)
    
    # Store mean and rstd
    tl.store(Mean_ptr + row_idx, mean)
    tl.store(Rstd_ptr + row_idx, rstd)
    
    # Apply normalization with weight and bias
    for col in range(0, n_cols, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(row_start_ptr + cols, mask=mask)
        w = tl.load(W_ptr + cols, mask=mask)
        b = tl.load(B_ptr + cols, mask=mask)
        
        y = (x - mean) * rstd
        y = y * w + b
        
        tl.store(Y_ptr + row_idx * stride + cols, y, mask=mask)

@triton.jit
def _layer_norm_backward_kernel(
    dY_ptr, X_ptr, Mean_ptr, Rstd_ptr, W_ptr,
    dX_ptr, dW_ptr, dB_ptr,
    stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    # Load mean and rstd for this row
    mean = tl.load(Mean_ptr + row_idx)
    rstd = tl.load(Rstd_ptr + row_idx)
    
    # Initialize accumulators
    sum_dy = 0.0
    sum_dy_xmu = 0.0
    
    # First pass: compute sum(dy) and sum(dy * (x-μ))
    row_start_x = X_ptr + row_idx * stride
    row_start_dy = dY_ptr + row_idx * stride
    
    for col in range(0, n_cols, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        dy = tl.load(row_start_dy + cols, mask=mask, other=0.0)
        x = tl.load(row_start_x + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0)
        
        xmu = (x - mean) * rstd
        dy_w = dy * w
        
        sum_dy += tl.sum(dy_w, mask=mask)
        sum_dy_xmu += tl.sum(dy_w * xmu, mask=mask)
    
    # Second pass: compute gradients
    for col in range(0, n_cols, BLOCK_SIZE):
        cols = col + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        dy = tl.load(row_start_dy + cols, mask=mask)
        x = tl.load(row_start_x + cols, mask=mask)
        w = tl.load(W_ptr + cols, mask=mask)
        
        xmu = (x - mean) * rstd
        dy_w = dy * w
        
        # Compute dx
        dx = (dy_w - (sum_dy + sum_dy_xmu * xmu) / n_cols) * rstd
        tl.store(dX_ptr + row_idx * stride + cols, dx, mask=mask)
        
        # Compute dw and db
        dw = dy * xmu
        db = dy
        
        # Atomic add for weight and bias gradients
        tl.atomic_add(dW_ptr + cols, dw, mask=mask)
        tl.atomic_add(dB_ptr + cols, db, mask=mask)

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        # Ensure inputs are contiguous
        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        
        # Get dimensions
        n_rows, n_cols = x.shape
        
        # Initialize output tensors
        y = torch.empty_like(x)
        mean = torch.empty(n_rows, dtype=x.dtype, device=x.device)
        rstd = torch.empty(n_rows, dtype=x.dtype, device=x.device)
        
        # Configure grid and block sizes
        BLOCK_SIZE = 128
        grid = (n_rows,)
        
        # Launch forward kernel
        _layer_norm_forward_kernel[grid](
            x.data_ptr(), weight.data_ptr(), bias.data_ptr(),
            y.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
            x.stride(0), n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        
        # Save for backward
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.n_cols = n_cols
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, mean, rstd = ctx.saved_tensors
        n_cols = ctx.n_cols
        
        # Initialize gradient tensors
        grad_input = torch.empty_like(x)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(weight)
        
        # Configure grid and block sizes
        BLOCK_SIZE = 128
        grid = (x.shape[0],)
        
        # Launch backward kernel
        _layer_norm_backward_kernel[grid](
            grad_output.data_ptr(), x.data_ptr(),
            mean.data_ptr(), rstd.data_ptr(), weight.data_ptr(),
            grad_input.data_ptr(), grad_weight.data_ptr(), grad_bias.data_ptr(),
            x.stride(0), n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        
        return grad_input, grad_weight, grad_bias, None

# Convenience wrapper
class TritonLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        
    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
