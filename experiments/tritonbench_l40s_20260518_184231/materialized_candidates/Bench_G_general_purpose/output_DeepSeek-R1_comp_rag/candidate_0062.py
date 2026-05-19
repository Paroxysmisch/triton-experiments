import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.jit
def _layer_norm_fwd_quant_kernel(
    X, Y, W, B, RESIDUAL, RESIDUAL_OUT, Mean, Rstd,
    stride_x_row, stride_y_row, stride_res_row, stride_res_out_row,
    N, eps, IS_RMS_NORM: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr, STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
        
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=mask, other=0.0).to(tl.float32)
        x += residual
    
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=mask)

    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = x - mean
    else:
        xbar = x
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    x_hat = xbar * rstd if not IS_RMS_NORM else x * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32) if HAS_WEIGHT else 1.0
    b = tl.load(B + cols, mask=mask).to(tl.float32) if HAS_BIAS else 0.0
    y = x_hat * w + b

    # Quantization logic
    abs_max = tl.max(tl.abs(y), axis=0)
    scale = 127.0 / tl.maximum(abs_max, 1e-5)
    y_quant = tl.math.round(y * scale)
    y_quant = tl.clip(y_quant, -128, 127)
    y_dequant = y_quant / scale

    tl.store(Y + cols, y_dequant, mask=mask)

def layer_norm_quant_forward(
    x, weight, bias, eps=1e-5, residual=None,
    out_dtype=None, is_rms_norm=False
):
    M, N = x.shape
    y = torch.empty_like(x, dtype=out_dtype or x.dtype)
    residual_out = None
    if residual is not None or (residual is None and out_dtype != x.dtype):
        residual_out = torch.empty(M, N, device=x.device, dtype=residual.dtype if residual is not None else x.dtype)
    
    mean = torch.empty(M, device=x.device, dtype=torch.float32) if not is_rms_norm else None
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise ValueError("Feature dimension exceeds maximum supported size")
    
    grid = (M,)
    _layer_norm_fwd_quant_kernel[grid](
        x, y, weight, bias, residual, residual_out, mean, rstd,
        x.stride(0), y.stride(0),
        residual.stride(0) if residual is not None else 0,
        residual_out.stride(0) if residual_out is not None else 0,
        N, eps, is_rms_norm, BLOCK_N,
        residual is not None, residual_out is not None,
        weight is not None, bias is not None
    )
    return y, mean, rstd, residual_out

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.jit
def _layer_norm_bwd_quant_kernel(
    X, W, B, Y, DY, DX, DW, DB, DRESIDUAL, DRESIDUAL_IN,
    Mean, Rstd, stride_x_row, stride_y_row, stride_dy_row,
    stride_dx_row, stride_dres_row, stride_dres_in_row,
    M, N, eps, rows_per_program, IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr, HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr, HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr, RECOMPUTE_OUTPUT: tl.constexpr
):
    row_block = tl.program_id(0)
    row_start = row_block * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    offs = row_start * stride_x_row
    X += offs
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32) if HAS_WEIGHT else None
    db = tl.zeros((BLOCK_N,), dtype=tl.float32) if HAS_BIAS else None
    
    for row in range(row_start, min(row_start + rows_per_program, M)):
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        
        x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32) if HAS_WEIGHT else 1.0
        
        if RECOMPUTE_OUTPUT:
            y = x_hat * w + (tl.load(B + cols, mask=mask, other=0.0) if HAS_BIAS else 0.0)
            abs_max = tl.max(tl.abs(y), axis=0)
            scale = 127.0 / tl.maximum(abs_max, 1e-5)
            y_quant = tl.math.round(y * scale)
            y_quant = tl.clip(y_quant, -128, 127)
            tl.store(Y + cols, y_quant / scale, mask=mask)
        
        wdy = dy * w
        if HAS_WEIGHT:
            dw += dy * x_hat
        if HAS_BIAS:
            db += dy
        
        if IS_RMS_NORM:
            c1 = tl.sum(x_hat * wdy, axis=0) / N
            dx = (wdy - x_hat * c1) * rstd
        else:
            c1 = tl.sum(x_hat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (x_hat * c1 + c2)) * rstd
        
        if HAS_DRESIDUAL:
            dx += tl.load(DRESIDUAL + cols, mask=mask, other=0.0)
        
        tl.store(DX + cols, dx, mask=mask)
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        
        X += stride_x_row
        DY += stride_dy_row
        DX += stride_dx_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
    
    if HAS_WEIGHT:
        tl.store(DW + row_block * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block * N + cols, db, mask=mask)

def layer_norm_quant_backward(
    dy, x, weight, bias, mean, rstd, eps=1e-5,
    dresidual=None, is_rms_norm=False, recompute_output=False
):
    M, N = x.shape
    dx = torch.empty_like(x)
    dresidual_in = torch.empty_like(x) if dx.dtype != x.dtype else None
    y = torch.empty_like(dy) if recompute_output else None
    
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise ValueError("Feature dimension exceeds maximum supported size")
    
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    rows_per_program = math.ceil(M / sm_count)
    
    _dw = torch.zeros((sm_count, N), device=x.device, dtype=torch.float32) if weight is not None else None
    _db = torch.zeros((sm_count, N), device=x.device, dtype=torch.float32) if bias is not None else None
    
    grid = (sm_count,)
    _layer_norm_bwd_quant_kernel[grid](
        x, weight, bias, y, dy, dx, _dw, _db, dresidual, dresidual_in,
        mean, rstd, x.stride(0), 0, dy.stride(0), dx.stride(0),
        dresidual.stride(0) if dresidual else 0,
        dresidual_in.stride(0) if dresidual_in else 0,
        M, N, eps, rows_per_program, is_rms_norm, BLOCK_N,
        dresidual is not None, dresidual_in is not None,
        weight is not None, bias is not None, recompute_output
    )
    
    dw = _dw.sum(0).to(weight.dtype) if weight else None
    db = _db.sum(0).to(bias.dtype) if bias else None
    return dx, dw, db, dresidual_in if dresidual_in is not None else dx
