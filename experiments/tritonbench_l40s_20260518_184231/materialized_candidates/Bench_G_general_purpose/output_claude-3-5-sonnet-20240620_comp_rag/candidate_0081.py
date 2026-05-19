import torch
import triton
import triton.language as tl
import math

@triton.jit
def rmsnorm_fwd_kernel(
    X, Y, W, Rstd,
    stride_ml, stride_n,
    L, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for the current instance
    row = tl.program_id(0)
    batch = tl.program_id(1)
    
    # Calculate base index for current slice
    base_idx = row * stride_ml + batch * stride_n
    X_ptr = X + base_idx
    Y_ptr = Y + base_idx
    
    # Compute RMS for the current row
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    
    rms = tl.sqrt(tl.sum(acc) / N + eps)
    tl.store(Rstd + row * L + batch, rms)
    
    # Normalize and scale
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        y = (x / rms) * w
        tl.store(Y_ptr + cols, y, mask=mask)

@triton.jit
def rmsnorm_bwd_kernel(
    DY, X, W, Rstd,
    DX, DW,
    stride_ml, stride_n,
    L, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    row = tl.program_id(0)
    batch = tl.program_id(1)
    
    # Calculate base index
    base_idx = row * stride_ml + batch * stride_n
    X_ptr = X + base_idx
    DY_ptr = DY + base_idx
    DX_ptr = DX + base_idx
    
    # Load RMS
    rms = tl.load(Rstd + row * L + batch)
    
    # Initialize accumulators
    dx_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    dw_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # First pass: compute sum(dy * x) for later use
    dy_x_sum = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        dy = tl.load(DY_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        dy_x_sum += tl.sum(dy * x * w, mask=mask)
    
    # Second pass: compute gradients
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        dy = tl.load(DY_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        
        # Compute dx
        dx = (dy * w / rms) - (x * dy_x_sum / (N * rms * rms * rms))
        tl.store(DX_ptr + cols, dx, mask=mask)
        
        # Compute dw (accumulated atomically)
        dw = dy * x / rms
        tl.atomic_add(DW + cols, dw, mask=mask)

class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        # Save input for backward pass
        ctx.eps = eps
        ctx.save_for_backward(x, weight)
        
        # Get input dimensions
        M, L, N = x.shape
        # Allocate output
        y = torch.empty_like(x)
        # Allocate space for RMS values
        rstd = torch.empty((M, L), device=x.device, dtype=x.dtype)
        
        # Calculate grid and block sizes
        BLOCK_SIZE = min(triton.next_power_of_2(N), 1024)
        grid = (M, L)
        
        # Launch kernel
        rmsnorm_fwd_kernel[grid](
            x, y, weight, rstd,
            x.stride(0), x.stride(1),
            L, N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        
        ctx.rstd = rstd
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        rstd = ctx.rstd
        
        # Get dimensions
        M, L, N = x.shape
        
        # Allocate gradients
        grad_input = torch.empty_like(x)
        grad_weight = torch.zeros_like(weight)
        
        # Calculate grid and block sizes
        BLOCK_SIZE = min(triton.next_power_of_2(N), 1024)
        grid = (M, L)
        
        # Launch backward kernel
        rmsnorm_bwd_kernel[grid](
            grad_output, x, weight, rstd,
            grad_input, grad_weight,
            x.stride(0), x.stride(1),
            L, N, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        
        return grad_input, grad_weight, None

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    return RMSNorm.apply(x, weight, eps)
