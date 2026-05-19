import torch
import triton
import triton.language as tl
import math

@triton.jit
def _layer_norm_fwd_fused(
    X, W, B, Y, Mean, Rstd,
    stride_xm, stride_xn,
    stride_wn, stride_bn,
    stride_ym, stride_yn,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    # Position of elements processed by this program
    row_idx = tl.program_id(0)
    
    # Load the row into SRAM
    row_start_ptr = X + row_idx * stride_xm
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < stride_xn
    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / stride_xn
    
    # Compute variance
    x_mean = x - mean
    x_var = tl.sum(x_mean * x_mean, axis=0) / stride_xn
    rstd = 1 / tl.sqrt(x_var + eps)
    
    # Load weight and bias
    w = tl.load(W + col_offsets, mask=mask, other=0.0)
    b = tl.load(B + col_offsets, mask=mask, other=0.0)
    
    # Normalize and apply affine transform
    y = w * (x_mean * rstd) + b
    
    # Write output
    output_ptr = Y + row_idx * stride_ym
    tl.store(output_ptr + col_offsets, y, mask=mask)
    
    # Store mean and rstd for backward pass
    tl.store(Mean + row_idx, mean)
    tl.store(Rstd + row_idx, rstd)

@triton.jit
def _layer_norm_bwd_dx_fused(
    DY, X, Mean, Rstd, W, DX, DW, DB,
    stride_dym, stride_dyn,
    stride_xm, stride_xn,
    stride_wn, stride_dxm, stride_dxn,
    N_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Position of elements
    row_idx = tl.program_id(0)
    
    # Load data for this row
    row_start_ptr = X + row_idx * stride_xm
    dy_row_start_ptr = DY + row_idx * stride_dym
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < stride_dxn
    
    x = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
    dy = tl.load(dy_row_start_ptr + col_offsets, mask=mask, other=0.0)
    w = tl.load(W + col_offsets, mask=mask, other=0.0)
    mean = tl.load(Mean + row_idx)
    rstd = tl.load(Rstd + row_idx)
    
    # Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    
    dx = (wdy - (tl.sum(wdy, axis=0) + xhat * tl.sum(wdy * xhat, axis=0)) / stride_dxn) * rstd
    
    # Store dx
    dx_row_start_ptr = DX + row_idx * stride_dxm
    tl.store(dx_row_start_ptr + col_offsets, dx, mask=mask)
    
    # Accumulate dw and db
    dw = dy * xhat
    db = dy
    
    # Use locks for accumulation
    lock_id = row_idx % GROUP_SIZE_M
    while tl.atomic_cas(LOCKS + lock_id, 0, 1) == 1:
        pass
    
    tl.atomic_add(DW + col_offsets, dw, mask=mask)
    tl.atomic_add(DB + col_offsets, db, mask=mask)
    
    # Release lock
    tl.atomic_xchg(LOCKS + lock_id, 0)

@triton.jit
def _layer_norm_bwd_dwdb(
    DW, DB, FINAL_DW, FINAL_DB,
    hidden_size: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    col_idx = tl.program_id(0)
    
    if col_idx < hidden_size:
        # Sum up partial gradients
        dw_sum = tl.sum(tl.load(DW + col_idx + tl.arange(0, GROUP_SIZE_M) * hidden_size))
        db_sum = tl.sum(tl.load(DB + col_idx + tl.arange(0, GROUP_SIZE_M) * hidden_size))
        
        # Store final gradients
        tl.store(FINAL_DW + col_idx, dw_sum)
        tl.store(FINAL_DB + col_idx, db_sum)

class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        # Determine dimensions
        batch_size, hidden_size = x.shape
        
        # Allocate output and intermediate buffers
        y = torch.empty_like(x)
        mean = torch.empty(batch_size, dtype=x.dtype, device=x.device)
        rstd = torch.empty(batch_size, dtype=x.dtype, device=x.device)
        
        # Configure kernel parameters
        BLOCK_SIZE = triton.next_power_of_2(hidden_size)
        num_warps = 4 if BLOCK_SIZE > 512 else 2
        
        # Launch forward kernel
        grid = (batch_size,)
        _layer_norm_fwd_fused[grid](
            x, weight, bias, y, mean, rstd,
            x.stride(0), x.stride(1),
            weight.stride(0), bias.stride(0),
            y.stride(0), y.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            eps=eps,
            num_warps=num_warps
        )
        
        # Save for backward
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, mean, rstd = ctx.saved_tensors
        batch_size, hidden_size = grad_output.shape
        
        # Allocate gradient buffers
        grad_x = torch.empty_like(x)
        GROUP_SIZE_M = min(batch_size, 32)
        num_groups = (batch_size + GROUP_SIZE_M - 1) // GROUP_SIZE_M
        
        grad_weight_partial = torch.zeros((num_groups, hidden_size), 
                                        dtype=x.dtype, device=x.device)
        grad_bias_partial = torch.zeros((num_groups, hidden_size), 
                                      dtype=x.dtype, device=x.device)
        
        # Launch backward kernels
        grid = (batch_size,)
        _layer_norm_bwd_dx_fused[grid](
            grad_output, x, mean, rstd, weight,
            grad_x, grad_weight_partial, grad_bias_partial,
            grad_output.stride(0), grad_output.stride(1),
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            N_ROWS=batch_size,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=ctx.num_warps
        )
        
        # Allocate final gradient buffers
        grad_weight = torch.empty_like(weight)
        grad_bias = torch.empty_like(weight)
        
        # Launch final reduction kernel
        grid = (hidden_size,)
        _layer_norm_bwd_dwdb[grid](
            grad_weight_partial, grad_bias_partial,
            grad_weight, grad_bias,
            hidden_size=hidden_size,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=1
        )
        
        return grad_x, grad_weight, grad_bias, None

# Helper function to create LayerNorm module
def layer_norm(x, weight, bias, eps=1e-5):
    return LayerNorm.apply(x, weight, bias, eps)
