import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn import functional as F

MAX_FUSED_SIZE = 65536  # max block size for kernel inputs

def calculate_settings(n):
    # Determine optimal block size and number of warps
    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n))
    num_warps = 4 if block_size <= 256 else 8
    return block_size, num_warps

@triton.jit
def layernorm_forward(
    x_ptr, y_ptr, w_ptr, b_ptr, r_ptr, mu_ptr, stride_x, stride_y, n_cols,
    eps, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Load input data pointers
    x_ptr += row_idx * stride_x
    y_ptr += row_idx * stride_y
    
    # Load weights and biases
    w = tl.load(w_ptr + cols, mask=cols < n_cols, other=0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=cols < n_cols, other=0).to(tl.float32)
    
    # Compute mean
    sum = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + cols, mask=mask, other=0).to(tl.float32)
        sum += tl.sum(x, axis=0)
    mean = sum / n_cols
    
    # Compute variance
    var_sum = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + cols, mask=mask, other=0).to(tl.float32)
        x_centered = x - mean
        var_sum += tl.sum(x_centered * x_centered, axis=0)
    var = var_sum / n_cols
    
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Write statistics for backward pass
    tl.store(mu_ptr + row_idx, mean)
    tl.store(r_ptr + row_idx, rstd)
    
    # Normalize and scale+shift
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + cols, mask=mask).to(tl.float32)
        x_norm = (x - mean) * rstd
        y = x_norm * w + b
        tl.store(y_ptr + cols, y, mask=mask)

@triton.jit
def layernorm_backward(
    dy_ptr, x_ptr, w_ptr, r_ptr, mu_ptr, dx_ptr, dw_ptr, db_ptr,
    stride_dy, stride_x, stride_dx, n_cols, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Load data pointers
    x_ptr += row_idx * stride_x
    dy_ptr += row_idx * stride_dy
    dx_ptr += row_idx * stride_dx
    
    # Load statistics
    mean = tl.load(mu_ptr + row_idx)
    rstd = tl.load(r_ptr + row_idx)
    
    # Load weights
    w = tl.load(w_ptr + cols, mask=cols < n_cols, other=0).to(tl.float32)
    
    # Compute intermediate values
    x_hat_sum = 0.0
    dy_sum = 0.0
    dy_x_hat_sum = 0.0
    
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        x = tl.load(x_ptr + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(dy_ptr + cols, mask=mask, other=0).to(tl.float32)
        x_hat = (x - mean) * rstd
        
        dy_sum += tl.sum(dy, axis=0)
        dy_x_hat = dy * x_hat
        dy_x_hat_sum += tl.sum(dy_x_hat, axis=0)
        x_hat_sum += tl.sum(x_hat, axis=0)
        
        # Compute dx
        dx_common = (dy * w) * rstd
        dx = dx_common - (x_hat * dy_x_hat_sum + dy_sum) / (n_cols * rstd)
        tl.store(dx_ptr + cols, dx, mask=mask)
    
    # Store gradients for weights and biases
    if dw_ptr is not None:
        tl.atomic_add(dw_ptr + cols, dy_x_hat, mask=mask)
    if db_ptr is not None:
        tl.atomic_add(db_ptr + cols, dy, mask=mask)

class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight, bias, eps):
        # Allocate output tensors
        y = torch.empty_like(x)
        n_cols = x.size(-1)
        
        # Allocate statistics storage
        r = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
        mu = torch.empty_like(r)
        
        # Calculate kernel settings
        block_size, num_warps = calculate_settings(n_cols)
        
        # Launch kernel
        grid = (x.numel() // n_cols,)
        layernorm_forward[grid](
            x, y, weight, bias, r, mu, x.stride(0), y.stride(0), n_cols, eps,
            BLOCK_SIZE=block_size, num_warps=num_warps
        )
        
        ctx.save_for_backward(x, weight, r, mu)
        ctx.n_cols = n_cols
        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        x, weight, r, mu = ctx.saved_tensors
        n_cols = ctx.n_cols
        
        # Initialize gradients
        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight)
        db = torch.zeros_like(weight)
        
        block_size, num_warps = calculate_settings(n_cols)
        grid = (x.size(0),)
        
        layernorm_backward[grid](
            dy, x, weight, r, mu, dx, dw, db, dy.stride(0), x.stride(0),
            dx.stride(0), n_cols, BLOCK_SIZE=block_size, num_warps=num_warps
        )
        return dx, dw, db, None

def fast_layernorm(x, module, eps=1e-5):
    weight = module.weight if module.elementwise_affine else None
    bias = module.bias if module.elementwise_affine else None
    return Fast_Layernorm.apply(x, weight, bias, eps)

layer = torch.nn.LayerNorm(hidden_size).cuda()
x = torch.randn(batch_size, hidden_size).cuda()

# Use the custom implementation
y = fast_layernorm(x, layer)
