import torch
import triton
import triton.language as tl
from triton import compile
from packaging import version

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

if TRITON3:
    from triton.language.extra.cuda.libdevice import rsqrt as _rsqrt
else:
    from triton.language.math import rsqrt as _rsqrt


@triton.jit
def _layer_norm_forward_kernel(
    X, Y, W, B, Mean, RSTD,
    stride_x_row, stride_y_row, stride_x_col, stride_y_col,
    N, eps,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE_COL)
    mask = col_offs < N
    row_begin = row_idx * stride_x_row
    row_x_begin = row_begin + tl.arange(0, BLOCK_SIZE_ROW) * stride_x_col
    row_y_begin = row_begin + tl.arange(0, BLOCK_SIZE_ROW) * stride_y_col
    x = tl.load(row_x_begin, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_zm = x - mean
    x_zm = tl.where(mask, x_zm, 0.)
    var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Mean + row_idx, mean)
    tl.store(RSTD + row_idx, rstd)
    mask = col_offs < N
    w = tl.load(W + col_offs, mask=mask).to(tl.float32)
    b = tl.load(B + col_offs, mask=mask).to(tl.float32)
    x_norm = x_zm * rstd * w + b
    y = x_norm.to(tl.float16)
    tl.store(row_y_begin, y, mask=mask)


@triton.jit
def _layer_norm_backward_kernel(
    DX, DW, DB,
    X, W, B, Mean, RSTD,
    DY, DW_PART, DB_PART,
    stride_x_row, stride_y_row, stride_x_col, stride_y_col,
    N, eps,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    num_warps: tl.constexpr,
    sm_count: tl.constexpr
):
    row_offs = tl.arange(0, BLOCK_SIZE_ROW)
    col_idx = tl.program_id(0)
    col_begin = col_idx * stride_x_col
    row_x_begin = col_begin + row_offs * stride_x_row
    row_y_begin = col_begin + row_offs * stride_y_row
    mask = row_offs < N
    x = tl.load(row_x_begin, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(Mean + col_idx)
    rstd = tl.load(RSTD + col_idx)
    w = tl.load(W + col_idx).to(tl.float32)
    dy = tl.load(row_y_begin, mask=mask, other=0.0).to(tl.float32)
    x_hat = (x - mean) * rstd
    wdy = w * dy
    x_mu = x - mean
    x_mu = tl.where(mask, x_mu, 0.)
    x_mu_wdy = x_mu * wdy
    dl = tl.sum(x_mu_wdy, axis=0) / N
    d_w = x_hat * dy
    d_b = dy
    d_x = (wdy + (x_hat - dl * x_mu))
    tl.store(row_y_begin, dy, mask=mask)
    tl.store(row_x_begin, d_x, mask=mask)
    d_w = tl.sum(d_w, axis=0)
    d_b = tl.sum(d_b, axis=0)
    d_w = d_w.to(tl.float16)
    d_b = d_b.to(tl.float16)
    d_w_part_offs = tl.arange(0, num_warps) * BLOCK_SIZE_COL + col_idx
    d_b_part_offs = tl.arange(0, num_warps) * BLOCK_SIZE_COL + col_idx + N
    tl.store(DW_PART + d_w_part_offs, d_w)
    tl.store(DB_PART + d_b_part_offs, d_b)


def layer_norm_forward(x, weight, bias, eps):
    shape = x.shape
    dim = shape[-1]
    x = x.reshape(-1, dim)
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
    BLOCK_SIZE_ROW = triton.next_power_of_2(x.shape[0])
    BLOCK_SIZE_COL = triton.next_power_of_2(max(1, x.shape[-1] // 2))
    num_warps = min(max(BLOCK_SIZE_COL // 256, 1), 8)
    grid = (x.shape[0], )
    x_col_stride = x.stride(0)
    x_row_stride = x.stride(1)
    y_col_stride = y.stride(0)
    y_row_stride = y.stride(1)
    _layer_norm_forward_kernel[grid](
        x, y, weight, bias, mean, rstd,
        x_row_stride, y_row_stride, x_col_stride, y_col_stride,
        N, eps,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
        num_warps=num_warps,
        # 1 sm is enough for this kernel
        # TODO: auto-detect this
        sm_count=1
    )
    y = y.reshape(shape)
    return y, mean, rstd


def layer_norm_backward(dy, x, weight, mean, rstd, eps, dw_part_sum_shape, db_part_sum_shape):
    shape = x.shape
    dim = shape[-1]
    x = x.reshape(-1, dim)
    M, N = x.shape
    dx = torch.empty_like(x)
    dw = torch.empty(dw_part_sum_shape, dtype=x.dtype, device=x.device)
    db = torch.empty(db_part_sum_shape, dtype=x.dtype, device=x.device)
    BLOCK_SIZE_ROW = triton.next_power_of_2(x.shape[0])
    BLOCK_SIZE_COL = triton.next_power_of_2(max(1, x.shape[-1] // 2))
    num_warps = min(max(BLOCK_SIZE_COL //
