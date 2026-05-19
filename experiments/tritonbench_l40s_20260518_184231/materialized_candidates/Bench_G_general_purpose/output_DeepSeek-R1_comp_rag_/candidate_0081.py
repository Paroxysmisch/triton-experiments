import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_fwd_kernel(
    X, Y, W, B, Mean, Rstd,
    stride_row, stride_feature,
    N, eps,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    X += row * stride_row
    Y += row * stride_row
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute sum and sum of squares
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += x
        _sum_sq += x * x
    
    sum_val = tl.sum(_sum, axis=0)
    sum_sq_val = tl.sum(_sum_sq, axis=0)
    mean = sum_val / N
    var = (sum_sq_val / N) - (mean * mean)
    rstd = 1.0 / tl.sqrt(var + eps)
    
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    
    # Normalize and output
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def layer_norm_bwd_dx_kernel(
    X, DY, W, Mean, Rstd, DX, 
    DW_partial, DB_partial, locks,
    stride_row, stride_feature,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    num_rows = num_pid_m * GROUP_SIZE_M
    
    row = pid // num_pid_n * GROUP_SIZE_M
    rows = row + tl.arange(0, GROUP_SIZE_M)
    row_mask = rows < num_rows
    
    for row_idx in rows:
        if row_mask[row_idx]:
            mean = tl.load(Mean + row_idx)
            rstd = tl.load(Rstd + row_idx)
            X_row = X + row_idx * stride_row
            DY_row = DY + row_idx * stride_row
            DX_row = DX + row_idx * stride_row
            
            sum_dy = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            sum_dy_xhat = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            for off in range(0, N, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                mask = cols < N
                x = tl.load(X_row + cols, mask=mask, other=0.0)
                dy = tl.load(DY_row + cols, mask=mask, other=0.0)
                w = tl.load(W + cols, mask=mask)
                x_hat = (x - mean) * rstd
                dy_w = dy * w
                sum_dy += dy_w
                sum_dy_xhat += dy_w * x_hat
            
            sum_dy_total = tl.sum(sum_dy, axis=0)
            sum_dy_xhat_total = tl.sum(sum_dy_xhat, axis=0)
            
            for off in range(0, N, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                mask = cols < N
                x = tl.load(X_row + cols, mask=mask, other=0.0)
                dy = tl.load(DY_row + cols, mask=mask, other=0.0)
                w = tl.load(W + cols, mask=mask)
                x_hat = (x - mean) * rstd
                dy_w = dy * w
                dx = (dy_w - (sum_dy_total/N + x_hat*sum_dy_xhat_total/N)) * rstd
                tl.store(DX_row + cols, dx, mask=mask)
                
                # Accumulate partial gradients
                partial_dw = dy * x_hat
                partial_db = dy
                lock_id = cols % locks.shape[0]
                for i in range(BLOCK_SIZE):
                    if mask[i]:
                        tl.atomic_add(DW_partial + cols[i], partial_dw[i])
                        tl.atomic_add(DB_partial + cols[i], partial_db[i])

@triton.jit
def layer_norm_bwd_dwdb_kernel(
    DW_partial, DB_partial, FINAL_DW, FINAL_DB, N,
    BLOCK_SIZE: tl.constexpr
):
    col = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col < N
    dw = tl.load(DW_partial + col, mask=mask, other=0.0)
    db = tl.load(DB_partial + col, mask=mask, other=0.0)
    tl.store(FINAL_DW + col, dw, mask=mask)
    tl.store(FINAL_DB + col, db, mask=mask)

class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        M, N = x.shape[:-1], x.shape[-1]
        x_ = x.reshape(-1, N)
        y = torch.empty_like(x_)
        mean = torch.empty(x_.shape[0], dtype=torch.float32, device=x.device)
        rstd = torch.empty(x_.shape[0], dtype=torch.float32, device=x.device)
        
        BLOCK_SIZE = max(triton.next_power_of_2(N), 1024)
        def grid(meta): return (x_.shape[0],)
        layer_norm_fwd_kernel[grid](x_, y, weight, bias, mean, rstd,
                                   x_.stride(0), 1, N, eps, BLOCK_SIZE)
        
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y.reshape(*M, N)
    
    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, rstd = ctx.saved_tensors
        M, N = x.shape[:-1], x.shape[-1]
        x_ = x.reshape(-1, N)
        dy_ = dy.reshape(-1, N)
        
        dx = torch.empty_like(x_)
        dw = torch.zeros_like(weight)
        db = torch.zeros_like(weight)
        locks = torch.zeros(1024, dtype=torch.int32, device=x.device)
        
        GROUP_SIZE_M = 4
        BLOCK_SIZE = ctx.BLOCK_SIZE
        num_rows = x_.shape[0]
        grid = lambda meta: (triton.cdiv(num_rows, meta['GROUP_SIZE_M']),)
        
        layer_norm_bwd_dx_kernel[grid](
            x_, dy_, weight, mean, rstd, dx, dw, db, locks,
            x_.stride(0), 1, N, 1e-5, BLOCK_SIZE, GROUP_SIZE_M
        )
        
        layer_norm_bwd_dwdb_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
            dw, db, dw, db, N, BLOCK_SIZE
        )
        
        return dx.reshape(*x.shape), dw, db, None

# Usage
layer_norm = LayerNorm.apply
x = torch.randn(10, 512, device='cuda')
weight = torch.ones(512, device='cuda')
bias = torch.zeros(512, device='cuda')
y = layer_norm(x, weight, bias)
