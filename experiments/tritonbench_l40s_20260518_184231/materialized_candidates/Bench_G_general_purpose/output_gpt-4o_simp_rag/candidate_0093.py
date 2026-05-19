import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "HAS_RESIDUAL", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X, Y, W, B, RESIDUAL, Mean, Rstd,
    stride_x_row, stride_y_row, stride_res_row,
    N, eps, IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
    
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    
    mask = cols < N
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w if HAS_WEIGHT else x_hat
    if HAS_BIAS:
        y = y + b
    
    tl.store(Y + cols, y, mask=mask)


def layer_norm_fwd(x, weight, bias, eps, residual=None, is_rms_norm=False):
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((M,), dtype=torch.float32, device="cuda") if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
    
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[(M,)](
            x, y, weight, bias, residual, mean, rstd,
            x.stride(0), y.stride(0), residual.stride(0) if residual is not None else 0,
            N, eps, is_rms_norm, BLOCK_N,
            residual is not None, weight is not None, bias is not None
        )
    
    return y, mean, rstd


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "HAS_DRESIDUAL", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.jit
def _layer_norm_bwd_kernel(
    X, W, B, DY, DX, DW, DB, DRESIDUAL,
    Mean, Rstd, stride_x_row, stride_dy_row, stride_dx_row, stride_dres_row,
    M, N, eps, rows_per_program,
    IS_RMS_NORM: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr, HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr
):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        
        wdy = dy
        if HAS_WEIGHT:
            wdy = dy * w
            dw += dy * xhat
        if HAS_BIAS:
            db += dy
        
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        
        tl.store(DX + cols, dx, mask=mask)
        
        X += stride_x_row
        DY += stride_dy_row
        DX += stride_dx_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
    
    if HAS_WEIGHT:
        tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)


def layer_norm_bwd(dy, x, weight, bias, eps, mean, rstd, dresidual=None, has_residual=False, is_rms_norm=False):
    M, N = x.shape
    dx = torch.empty_like(x)
    y = torch.empty(M, N, dtype=dy.dtype, device=dy.device)
    
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device) if weight is not None else None
    _db = torch.empty((sm_count, N), dtype=torch.float32, device=bias.device) if bias is not None else None
    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    
    with torch.cuda.device(x.device.index):
        _layer_norm_bwd_kernel[grid](
            x, weight, bias, dy, dx, _dw, _db, dresidual,
            mean, rstd, x.stride(0), dy.stride(0), dx.stride(0),
            dresidual.stride(0) if dresidual is not None else 0,
            M, N, eps,
