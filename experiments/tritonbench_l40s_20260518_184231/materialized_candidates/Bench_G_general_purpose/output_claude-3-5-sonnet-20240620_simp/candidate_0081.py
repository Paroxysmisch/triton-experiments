import torch
import triton
import triton.language as tl
import math

@triton.jit
def _layer_norm_fwd_fused(
    X, W, B, Y, Mean, Rstd,
    stride_xm, stride_xn,
    stride_wn, stride_bn,
    stride_yn, BLOCK_M: tl.constexpr,
    N: tl.constexpr, eps: tl.constexpr
):
    # Position of elements processed by this program
    row = tl.program_id(0)
    
    # Offsets for this row
    offs_x = row * stride_xm + tl.arange(0, N) * stride_xn
    offs_w = tl.arange(0, N) * stride_wn
    offs_b = tl.arange(0, N) * stride_bn
    offs_y = row * stride_xm + tl.arange(0, N) * stride_yn
    
    # Load data
    x = tl.load(X + offs_x)
    w = tl.load(W + offs_w)
    b = tl.load(B + offs_b)
    
    # Compute mean
    mean = tl.sum(x) / N
    # Center data
    xc = x - mean
    # Compute variance
    var = tl.sum(xc * xc) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize
    y = xc * rstd
    # Scale and shift
    y = y * w + b
    
    # Write output
    tl.store(Y + offs_y, y)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

@triton.jit
def _layer_norm_bwd_dx_fused(
    DY, X, W, Mean, Rstd, DX, DW, DB,
    stride_dym, stride_dyn, stride_xm, stride_xn,
    stride_wn, stride_dxm, stride_dxn,
    M: tl.constexpr, N: tl.constexpr
):
    # Position
    row = tl.program_id(0)
    
    # Offsets
    offs_dy = row * stride_dym + tl.arange(0, N) * stride_dyn
    offs_x = row * stride_xm + tl.arange(0, N) * stride_xn
    offs_w = tl.arange(0, N) * stride_wn
    offs_dx = row * stride_dxm + tl.arange(0, N) * stride_dxn
    
    # Load data
    dy = tl.load(DY + offs_dy)
    x = tl.load(X + offs_x)
    w = tl.load(W + offs_w)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    
    # Compute centered input
    xc = (x - mean) * rstd
    
    # Compute gradients
    dw = dy * xc
    db = dy
    
    # Accumulate weight and bias gradients atomically
    tl.atomic_add(DW + tl.arange(0, N), dw)
    tl.atomic_add(DB + tl.arange(0, N), db)
    
    # Compute dx
    dx = dy * w * rstd
    dx = dx - tl.sum(dx) / N
    dx = dx - xc * tl.sum(dy * xc) / N
    
    # Store dx
    tl.store(DX + offs_dx, dx)

@triton.jit
def _layer_norm_bwd_dwdb(
    DW, DB, FINAL_DW, FINAL_DB,
    M: tl.constexpr, N: tl.constexpr
):
    # Position
    col = tl.program_id(0)
    
    # Load and sum gradients
    dw = tl.load(DW + col)
    db = tl.load(DB + col)
    
    # Store final gradients
    tl.store(FINAL_DW + col, dw)
    tl.store(FINAL_DB + col, db)

class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        # Determine launch grid
        M, N = x.shape
        # Allocate output
        y = torch.empty_like(x)
        mean = torch.empty((M,), dtype=x.dtype, device=x.device)
        rstd = torch.empty((M,), dtype=x.dtype, device=x.device)
        
        # Launch kernel
        grid = (M,)
        _layer_norm_fwd_fused[grid](
            x, self.weight, self.bias, y, mean, rstd,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.bias.stride(0),
            y.stride(1), BLOCK_M=32,
            N=N, eps=self.eps
        )
        
        ctx = (x, self.weight, mean, rstd)
        return y, ctx
    
    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, rstd = ctx
        M, N = dy.shape
        
        # Allocate output
        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight)
        db = torch.zeros_like(weight)
        final_dw = torch.empty_like(weight)
        final_db = torch.empty_like(weight)
        
        # Launch kernels
        grid = (M,)
        _layer_norm_bwd_dx_fused[grid](
            dy, x, weight, mean, rstd, dx, dw, db,
            dy.stride(0), dy.stride(1),
            x.stride(0), x.stride(1),
            weight.stride(0), dx.stride(0), dx.stride(1),
            M=M, N=N
        )
        
        grid = (N,)
        _layer_norm_bwd_dwdb[grid](
            dw, db, final_dw, final_db,
            M=M, N=N
        )
        
        return dx, final_dw, final_db
