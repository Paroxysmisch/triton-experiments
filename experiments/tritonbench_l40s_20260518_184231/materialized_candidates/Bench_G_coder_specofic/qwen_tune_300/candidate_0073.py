import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=8),
    ],
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X, Y, W, B, RESIDUAL, RESIDUAL_OUT, Mean, Rstd,
    stride_x_row, stride_y_row, stride_res_row, stride_res_out_row,
    N, eps, IS_RMS_NORM: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr, STORE_RESIDUAL_OUT: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
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
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    tl.store(Y + cols, y, mask=mask)


def _layer_norm_fwd(
    x, weight, bias, eps, residual=None, out_dtype=None, residual_dtype=None, is_rms_norm=False
):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    assert x.stride(-1) == 1
    if residual is not None:
        assert residual.stride(-1) == 1
        assert residual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    assert y.stride(-1) == 1
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(M, N, device=x.device, dtype=residual_dtype)
        assert residual_out.stride(-1) == 1
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float32, device="cuda") if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[(M,)](
            x, y, weight, bias, residual, residual_out, mean, rstd,
            x.stride(0), y.stride(0), residual.stride(0) if residual is not None else 0,
            residual_out.stride(0) if residual_out is not None else 0,
            N, eps, is_rms_norm, BLOCK_N, residual is not None, residual_out is not None, bias is not None,
        )
    return y, mean, rstd, residual_out if residual_out is not None else x

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=8),
    ],
    key=["N", "HAS_DRESIDUAL", "RECOMPUTE_OUTPUT", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.jit
def _layer_norm_bwd_kernel(
    X, W, B, Y, DY, DX, DW, DB, DRESIDUAL, DRESIDUAL_IN,
    Mean, Rstd, stride_x_row, stride_y_row, stride_dy_row, stride_dx_row,
    stride_dres_row, stride_dres_in_row, N, eps, rows_per_program,
    IS_RMS_NORM: tl.constexpr, BLOCK_N: tl.constexpr, HAS_DRESIDUAL: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if HAS_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if RECOMPUTE_OUTPUT:
        y = tl.load(Y + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, N)
    for row in range(row_start, row_end):
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            yhat = xhat * w + b if HAS_BIAS else xhat * w
            tl.store(Y + cols, yhat, mask=mask)
        wdy = w * dy
        dw += dy * xhat
        db += dy * (xhat * wdy if not IS_RMS_NORM else wdy)
        if HAS_DRESIDUAL:
            residual = tl.load(DRESIDUAL + cols, mask=mask, other=0.0).to(tl.float32)
            x += residual
            tl.store(DRESIDUAL_IN + cols, x, mask=mask)
        dx = (wdy - (tl.sum(wdy, axis=0) / N)) * rstd if not IS_RMS_NORM else wdy * rstd
        tl.store(DX + cols, dx, mask=mask)
    if HAS_BIAS:
        tl.store(DB + (row_block_id * BLOCK_N + cols), db, mask=mask)
    tl.store(DW + (row_block_id * BLOCK_N + cols), dw, mask=mask)


def _layer_norm_bwd(dout, x, weight, bias, eps, mean, rstd, residual=None, out_dtype=None, residual_dtype=None, is_rms_norm=False, recompute_output=False):
    M, N = x.shape
    assert dout.stride(-1) == 1
    x = x.contiguous()
    if residual is not None:
        assert residual.stride(-1) == 1
        residual = residual.contiguous()
    assert weight.stride
