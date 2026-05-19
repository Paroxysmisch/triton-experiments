import torch
import triton
import triton.language as tl

# Helper function to calculate optimal block size and warps
def calculate_settings(n_cols, element_size):
    MAX_FUSED_SIZE = 65536 // element_size
    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    num_warps = min(max(block_size // 256, 1), 8)
    return block_size, num_warps

# Forward kernel
@triton.jit
def layernorm_forward(
    x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr,
    stride_x, n_cols, eps, 
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Compute mean
    mean = 0.0
    x_ptrs = x_ptr + row_start + cols
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols_mask = cols + offset < n_cols
        x = tl.load(x_ptrs + offset, mask=cols_mask, other=0.0)
        mean += tl.sum(x, axis=0)
    mean /= n_cols
    
    # Compute variance
    var = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols_mask = cols + offset < n_cols
        x = tl.load(x_ptrs + offset, mask=cols_mask, other=0.0)
        x_centered = tl.where(cols_mask, x - mean, 0.0)
        var += tl.sum(x_centered * x_centered, axis=0)
    var /= n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store statistics
    tl.store(mean_ptr + row_idx, mean)
    tl.store(rstd_ptr + row_idx, rstd)
    
    # Normalize and transform
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols_mask = cols + offset < n_cols
        x = tl.load(x_ptrs + offset, mask=cols_mask)
        w = tl.load(w_ptr + cols + offset, mask=cols_mask)
        b = tl.load(b_ptr + cols + offset, mask=cols_mask)
        
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(y_ptr + row_start + offset, y, mask=cols_mask)

# Backward input gradient kernel
@triton.jit
def layernorm_backward_dx(
    dx_ptr, dy_ptr, dw_partial_ptr, db_partial_ptr,
    x_ptr, w_ptr, mean_ptr, rstd_ptr,
    lock_ptr, stride_x, n_cols,
    GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    # Load data
    x = tl.load(x_ptr + row_idx * stride_x + cols, mask=mask)
    dy = tl.load(dy_ptr + row_idx * stride_x + cols, mask=mask)
    w = tl.load(w_ptr + cols, mask=mask)
    mean = tl.load(mean_ptr + row_idx)
    rstd = tl.load(rstd_ptr + row_idx)
    
    # Compute gradients
    x_hat = (x - mean) * rstd
    wdy = w * dy
    c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
    c2 = tl.sum(wdy, axis=0) / n_cols
    dx = (wdy - (x_hat * c1 + c2)) * rstd
    tl.store(dx_ptr + row_idx * stride_x + cols, dx, mask=mask)
    
    # Accumulate partial weight/bias gradients
    partial_dw = dy * x_hat
    partial_db = dy
    
    # Atomic updates with locks
    lock_id = row_idx % GROUP_SIZE_M
    lock = lock_ptr + lock_id
    count_ptr = lock_ptr + GROUP_SIZE_M + lock_id
    
    while tl.atomic_cas(lock, 0, 1) == 1:
        pass
    
    current_count = tl.load(count_ptr)
    if current_count == 0:
        tl.store(count_ptr, 1)
    else:
        partial_dw += tl.load(dw_partial_ptr + lock_id * n_cols + cols, mask=mask)
        partial_db += tl.load(db_partial_ptr + lock_id * n_cols + cols, mask=mask)
    
    tl.store(dw_partial_ptr + lock_id * n_cols + cols, partial_dw, mask=mask)
    tl.store(db_partial_ptr + lock_id * n_cols + cols, partial_db, mask=mask)
    tl.atomic_xchg(lock, 0)

# Backward parameters gradient kernel
@triton.jit
def layernorm_backward_dwdb(
    dw_partial_ptr, db_partial_ptr,
    dw_ptr, db_ptr,
    group_size, n_cols,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cols = cols < n_cols
    
    dw_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for group_offset in range(0, group_size, BLOCK_SIZE_M):
        rows = group_offset + tl.arange(0, BLOCK_SIZE_M)
        mask_rows = rows < group_size
        
        addrs = rows[:, None] * n_cols + cols[None, :]
        mask = mask_rows[:, None] & mask_cols[None, :]
        
        dw_acc += tl.load(dw_partial_ptr + addrs, mask=mask, other=0.0)
        db_acc += tl.load(db_partial_ptr + addrs, mask=mask, other=0.0)
    
    dw_total = tl.sum(dw_acc, axis=0)
    db_total = tl.sum(db_acc, axis=0)
    
    tl.store(dw_ptr + cols, dw_total, mask=mask_cols)
    tl.store(db_ptr + cols, db_total, mask=mask_cols)

# Autograd Function
class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        y = torch.empty_like(x)
        x_2d = x.reshape(-1, x.shape[-1])
        n_rows, n_cols = x_2d.shape
        
        element_size = x.element_size()
        block_size, num_warps = calculate_settings(n_cols, element_size)
        
        mean = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        
        layernorm_forward[(n_rows,)](
            x_2d, y, weight, bias, mean, rstd,
            x_2d.stride(0), n_cols, eps,
            BLOCK_SIZE=block_size, num_warps=num_warps
        )
        
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.n_cols = n_cols
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        return y.reshape_as(x)

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        x, weight, bias, mean, rstd = ctx.saved_tensors
        
        # Determine group size
        GROUP_SIZE_M = 64
        if ctx.n_cols <= 1024:
            GROUP_SIZE_M = 256
        elif ctx.n_cols <= 4096:
            GROUP_SIZE_M = 128
        elif ctx.n_cols <= 8192:
            GROUP_SIZE_M = 96
        
        # Initialize gradients
        dx = torch.empty_like(dy)
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=dy.device)
        dw_partial = torch.zeros((GROUP_SIZE_M, ctx.n_cols), device=dy.device)
        db_partial = torch.zeros_like(dw_partial)
        
        # Compute input gradient
        layernorm_backward_dx[(x.shape[0],)](
            dx, dy, dw_partial, db_partial,
            x, weight, mean, rstd, locks,
            x.stride(-1), ctx.n_cols,
            GROUP_SIZE_M=GROUP_SIZE_M,
            BLOCK_SIZE=ctx.block_size,
            num_warps=ctx.num_warps
        )
        
        # Compute parameter gradients
        dw = torch.empty_like(weight)
        db = torch.empty_like(bias)
        grid = lambda meta: [triton.cdiv(ctx.n_cols, meta['BLOCK_SIZE_N'])]
        
        layernorm_backward_dwdb[grid](
            dw_partial, db_partial, dw, db,
            min(GROUP_SIZE_M, x.shape[0]), ctx.n_cols,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128
        )
        
        return dx, dw, db, None

# User-facing function
def fast_layernorm(x, ln_layer):
    if not ln_layer.elementwise_affine:
        return Fast_Layernorm.apply(x, None, None, ln_layer.eps)
    return Fast_Layernorm.apply(x, ln_layer.weight, ln_layer.bias, ln_layer.eps)

import torch.nn as nn

class TritonLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape)
        
    def forward(self, x):
        return fast_layernorm(x, self.ln)
