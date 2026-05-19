import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'GROUP_SIZE_M': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'GROUP_SIZE_M': 4}, num_warps=8),
    ],
    key=['N']
)
@triton.jit
def _layer_norm_fwd_fused(
    X, Y, W, B, Mean, Rstd,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Offset into current row
    X += row * stride
    Y += row * stride
    
    # Load data and weights
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0).to(tl.float32)
    
    # Compute mean and variance using parallel reduction
    mean = tl.sum(x, axis=0) / N
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and scale
    y = x_centered * rstd * w + b
    
    # Store results
    tl.store(Y + cols, y, mask=mask)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

@triton.jit
def _layer_norm_bwd_dx_fused(
    DY, DX, DW, DB, W, B, Mean, Rstd, Lock,
    stride, N, 
    BLOCK_SIZE: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Load data
    dy = tl.load(DY + row * stride + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    
    # Compute gradients
    xhat = (dy * w) * rstd
    c1 = tl.sum(xhat, axis=0) / N
    c2 = tl.sum(xhat * (cols * stride - mean) * rstd, axis=0) / N
    
    dx = xhat - c1 - (cols * stride - mean) * rstd * c2
    tl.store(DX + row * stride + cols, dx, mask=mask)
    
    # Partial sums for weights and biases
    partial_dw = tl.sum(dy * (cols * stride - mean) * rstd, axis=0)
    partial_db = tl.sum(dy, axis=0)
    
    # Accumulate partials using locks
    lock_id = row % GROUP_SIZE_M
    lock = Lock + lock_id
    with tl.atomic_add(lock, 0) as lock_handle:
        dw = tl.load(DW + lock_id * N + cols, mask=mask, other=0)
        db = tl.load(DB + lock_id * N + cols, mask=mask, other=0)
        tl.store(DW + lock_id * N + cols, dw + partial_dw, mask=mask)
        tl.store(DB + lock_id * N + cols, db + partial_db, mask=mask)

@triton.jit
def _layer_norm_bwd_dwdb(
    DW, DB, FINAL_DW, FINAL_DB,
    N, M, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    dw = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for i in range(M):
        dw += tl.load(DW + i * N + cols, mask=mask, other=0)
        db += tl.load(DB + i * N + cols, mask=mask, other=0)
    
    tl.store(FINAL_DW + cols, dw, mask=mask)
    tl.store(FINAL_DB + cols, db, mask=mask)

class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        # Validate inputs
        assert x.is_contiguous()
        assert weight.shape == x.shape[-1:]
        
        # Allocate output tensors
        y = torch.empty_like(x)
        mean = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
        rstd = torch.empty_like(mean)
        
        # Kernel configuration
        N = x.size(-1)
        config = {'BLOCK_SIZE': min(triton.next_power_of_2(N), 1024)}
        grid = (x.numel() // N,)
        
        # Launch kernel
        _layer_norm_fwd_fused[grid](
            x, y, weight, bias, mean, rstd,
            x.stride(0), N, eps,
            GROUP_SIZE_M=4, **config
        )
        
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, mean, rstd = ctx.saved_tensors
        dx, dw, db = None, None, None
        
        # Allocate gradients
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(x)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            M = x.size(0)
            N = x.size(-1)
            locks = torch.zeros(4, dtype=torch.int32, device=x.device)
            dw_partial = torch.zeros((4, N), dtype=x.dtype, device=x.device)
            db_partial = torch.zeros_like(dw_partial)
            
            # Launch DX kernel
            _layer_norm_bwd_dx_fused[(M,)](
                dy, dx, dw_partial, db_partial, w, b, mean, rstd, locks,
                x.stride(0), N,
                BLOCK_SIZE=128, GROUP_SIZE_M=4
            )
            
            # Launch reduction kernel
            _layer_norm_bwd_dwdb[(triton.cdiv(N, 256),)](
                dw_partial, db_partial, dw, db,
                N, 4, BLOCK_SIZE=256
            )
        
        return dx, dw, db, None

# Usage example
layer_norm = LayerNorm.apply
x = torch.randn(32, 1024, device='cuda')
w = torch.ones(1024, device='cuda')
b = torch.zeros(1024, device='cuda')
y = layer_norm(x, w, b, 1e-5)
