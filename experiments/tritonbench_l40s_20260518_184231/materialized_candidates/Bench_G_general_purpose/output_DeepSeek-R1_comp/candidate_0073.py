import torch
import triton
import triton.language as tl

# Forward Kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256, 'PRELOAD_WEIGHTS': False}, num_warps=4),
        triton.Config({'BLOCK_N': 512, 'PRELOAD_WEIGHTS': True}, num_warps=4),
        triton.Config({'BLOCK_N': 1024, 'PRELOAD_WEIGHTS': True}, num_warps=8),
    ],
    key=['n_cols']
)
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    # Tensors
    x_ptr, residual_ptr, weight_ptr, bias_ptr, 
    y_ptr, residual_out_ptr, mean_ptr, rstd_ptr,
    # Dimensions
    n_rows, n_cols,
    # Meta-parameters
    use_residual: tl.constexpr,
    store_residual: tl.constexpr,
    rms_norm: tl.constexpr,
    eps: tl.constexpr,
    residual_inplace: tl.constexpr,
    # Block config
    BLOCK_N: tl.constexpr,
    PRELOAD_WEIGHTS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    
    # Load input data
    x = tl.load(x_ptr + row * n_cols + cols, mask=cols < n_cols, other=0).to(tl.float32)
    
    # Add residual if needed
    if use_residual:
        residual = tl.load(residual_ptr + row * n_cols + cols, mask=cols < n_cols, other=0).to(tl.float32)
        if residual_inplace:
            x += residual
        else:
            x = x + residual

    # Compute mean and variance
    if rms_norm:
        mean = 0.0
        var = tl.sum(x * x, axis=0) / n_cols
    else:
        mean = tl.sum(x, axis=0) / n_cols
        var = tl.sum((x - mean) * (x - mean), axis=0) / n_cols
        
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and apply weights/bias
    x_hat = (x - mean) * rstd if not rms_norm else x * rstd
    
    if PRELOAD_WEIGHTS:
        weight = tl.load(weight_ptr + cols, mask=cols < n_cols)
        bias = tl.load(bias_ptr + cols, mask=(bias_ptr != 0) & (cols < n_cols)) if bias_ptr != 0 else 0
    else:
        weight = tl.load(weight_ptr + cols, mask=cols < n_cols, other=1.0)
        bias = tl.load(bias_ptr + cols, mask=(bias_ptr != 0) & (cols < n_cols), other=0.0) if bias_ptr != 0 else 0.0

    y = x_hat * weight + bias
    
    # Store outputs
    tl.store(y_ptr + row * n_cols + cols, y, mask=cols < n_cols)
    if store_residual:
        tl.store(residual_out_ptr + row * n_cols + cols, x, mask=cols < n_cols)
    if mean_ptr != 0:
        tl.store(mean_ptr + row, mean)
    if rstd_ptr != 0:
        tl.store(rstd_ptr + row, rstd)

# Backward Kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256, 'NUM_ROWS_PER_PROGRAM': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 512, 'NUM_ROWS_PER_PROGRAM': 256}, num_warps=8),
    ],
    key=['n_rows', 'n_cols']
)
@triton.jit
def _layer_norm_bwd_kernel(
    # Tensors
    dy_ptr, x_ptr, residual_ptr, weight_ptr, mean_ptr, rstd_ptr,
    dx_ptr, dresidual_ptr, dweight_ptr, dbias_ptr,
    # Dimensions
    n_rows, n_cols,
    # Parameters
    use_residual: tl.constexpr,
    has_residual_grad: tl.constexpr,
    rms_norm: tl.constexpr,
    needs_recompute: tl.constexpr,
    # Block config
    BLOCK_N: tl.constexpr,
    NUM_ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * NUM_ROWS_PER_PROGRAM
    rows = row_start + tl.arange(0, NUM_ROWS_PER_PROGRAM)
    cols = tl.arange(0, BLOCK_N)
    
    for i in range(NUM_ROWS_PER_PROGRAM):
        row = row_start + i
        if row >= n_rows:
            break
            
        # Load forward data
        if needs_recompute:
            x = tl.load(x_ptr + row * n_cols + cols, mask=cols < n_cols, other=0).to(tl.float32)
            if use_residual:
                residual = tl.load(residual_ptr + row * n_cols + cols, mask=cols < n_cols, other=0).to(tl.float32)
                x += residual
            mean = tl.sum(x, axis=0) / n_cols if not rms_norm else 0.0
            var = tl.sum((x - mean) * (x - mean), axis=0) / n_cols if not rms_norm else tl.sum(x * x, axis=0) / n_cols
            rstd = 1.0 / tl.sqrt(var + 1e-6)
        else:
            mean = tl.load(mean_ptr + row) if not rms_norm else 0.0
            rstd = tl.load(rstd_ptr + row)
        
        # Load gradients
        dy = tl.load(dy_ptr + row * n_cols + cols, mask=cols < n_cols, other=0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=cols < n_cols).to(tl.float32)
        
        # Compute gradients
        x_hat = (x - mean) * rstd if not rms_norm else x * rstd
        dl_dxhat = dy * weight
        dmean = tl.sum(-dl_dxhat * rstd, axis=0) if not rms_norm else 0.0
        drstd = tl.sum((x - mean) * dl_dxhat, axis=0) if not rms_norm else tl.sum(x * dl_dxhat, axis=0)
        
        # Input gradient
        dx = dl_dxhat * rstd + dmean / n_cols + drstd * (x - mean) * (-rstd ** 3) / n_cols if not rms_norm else dl_dxhat * rstd
        
        # Weight/Bias gradients
        dweight = tl.sum(dy * x_hat, axis=0)
        dbias = tl.sum(dy, axis=0)
        
        # Store gradients
        if dx_ptr != 0:
            tl.store(dx_ptr + row * n_cols + cols, dx, mask=cols < n_cols)
        if has_residual_grad:
            tl.store(dresidual_ptr + row * n_cols + cols, dx, mask=cols < n_cols)
        
        # Atomic updates for parameter gradients
        if dweight_ptr != 0:
            tl.atomic_add(dweight_ptr + cols, dweight, mask=cols < n_cols)
        if dbias_ptr != 0:
            tl.atomic_add(dbias_ptr + cols, dbias, mask=cols < n_cols)

# Wrapper Functions
def layer_norm_forward(x, weight, bias=None, residual=None, eps=1e-6, 
                       use_residual=False, store_residual=False, rms_norm=False):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    residual_out = torch.empty_like(x) if store_residual else None
    
    # Determine BLOCK_N based on hardware constraints
    MAX_FEATURES = 4096
    BLOCK_N = triton.next_power_of_2(min(n_cols, MAX_FEATURES))
    
    # Launch kernel
    grid = (n_rows,)
    _layer_norm_fwd_1pass_kernel[grid](
        x, residual, weight, bias,
        y, residual_out,
        None, None,  # mean and rstd not stored
        n_rows, n_cols,
        use_residual=use_residual,
        store_residual=store_residual,
        rms_norm=rms_norm,
        eps=eps,
        residual_inplace=(residual is not None),
        BLOCK_N=BLOCK_N,
    )
    return y, residual_out

def layer_norm_backward(dy, x, residual, weight, mean, rstd,
                        needs_recompute=False, use_residual=False, rms_norm=False):
    n_rows, n_cols = x.shape
    dx = torch.empty_like(x) if use_residual else None
    dresidual = torch.empty_like(x) if use_residual else None
    dweight = torch.zeros_like(weight)
    dbias = torch.zeros_like(weight) if weight.requires_grad else None
    
    # Configure SM-optimized grid
    num_SMs = torch.cuda.get_device_properties(x.device).multi_processor_count
    rows_per_SM = (n_rows + num_SMs - 1) // num_SMs
    
    # Launch kernel
    grid = (num_SMs,)
    _layer_norm_bwd_kernel[grid](
        dy, x, residual, weight, mean, rstd,
        dx, dresidual, dweight, dbias,
        n_rows, n_cols,
        use_residual=use_residual,
        has_residual_grad=(dresidual is not None),
        rms_norm=rms_norm,
        needs_recompute=needs_recompute,
        NUM_ROWS_PER_PROGRAM=rows_per_SM,
    )
    return dx, dweight, dbias, dresidual
