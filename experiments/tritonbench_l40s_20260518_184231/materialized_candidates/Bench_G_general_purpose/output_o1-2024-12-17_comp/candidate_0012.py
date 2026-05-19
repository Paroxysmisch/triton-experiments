import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_forward_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr, Mean_ptr, RSTD_ptr,
    stride_xi, stride_xj, stride_w, stride_b, stride_yi, stride_yj,
    stride_m, stride_r, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    # Load input, weight, and bias
    x = tl.load(X_ptr + offs * stride_xj, mask=mask, other=0.0)
    w = tl.load(W_ptr + offs * stride_w, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs * stride_b, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / n_cols
    # Compute variance and rstd
    var = tl.sum((x - mean) * (x - mean), axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # Normalize and apply weight/bias
    y = (x - mean) * rstd * w + b

    # Store results
    tl.store(Y_ptr + offs * stride_yj, y, mask=mask)
    # Single mean/rstd for the entire row
    if tl.thread_id_in_cluster(0) == 0:
        tl.store(Mean_ptr, mean)
        tl.store(RSTD_ptr, rstd)


@triton.jit
def _layer_norm_backward_kernel(
    X_ptr, W_ptr, B_ptr, DY_ptr, Mean_ptr, RSTD_ptr,
    DX_ptr, DW_ptr, DB_ptr,
    n_rows, n_cols,
    stride_xi, stride_xj, stride_w, stride_b,
    stride_dyi, stride_dyj, stride_dxi, stride_dxj,
    stride_dw, stride_db,
    stride_m, stride_r,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    cols = row_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Load mean and rstd (assume single value per row)
    mean = tl.load(Mean_ptr + row_id * stride_m)
    rstd = tl.load(RSTD_ptr + row_id * stride_r)

    # Load X and DY
    x = tl.load(X_ptr + row_id * stride_xi + cols * stride_xj, mask=mask, other=0.0)
    dy = tl.load(DY_ptr + row_id * stride_dyi + cols * stride_dyj, mask=mask, other=0.0)

    # Load W
    w = tl.load(W_ptr + cols * stride_w, mask=mask, other=0.0)

    # Grad wrt X
    x_hat = (x - mean) * rstd
    dw_part = x_hat * dy
    db_part = dy
    dx_hat = w * dy
    sum_dx_hat = tl.sum(dx_hat, axis=0)
    sum_x_hat_dx_hat = tl.sum(x_hat * dx_hat, axis=0)

    dx = (dx_hat - (sum_dx_hat / n_cols) - (x_hat * sum_x_hat_dx_hat / n_cols)) * rstd

    # Store partial DX
    tl.store(DX_ptr + row_id * stride_dxi + cols * stride_dxj, dx, mask=mask)

    # Reduction for W, B
    # We accumulate partial sums from each program; a second pass may be needed
    tl.atomic_add(DW_ptr + cols * stride_dw, dw_part, mask=mask)
    tl.atomic_add(DB_ptr + cols * stride_db, db_part, mask=mask)


def calculate_settings(n_cols):
    BLOCK_SIZE = 128
    num_warps = 4
    # Adjust BLOCK_SIZE / num_warps if needed based on n_cols
    return BLOCK_SIZE, num_warps


def layer_norm_forward(X, W, B):
    # X: (n_rows, n_cols)
    n_rows, n_cols = X.shape
    Y = torch.empty_like(X)
    Mean = torch.empty(n_rows, device=X.device, dtype=X.dtype)
    RSTD = torch.empty(n_rows, device=X.device, dtype=X.dtype)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    grid = lambda meta: (triton.cdiv(n_cols, BLOCK_SIZE),)
    _layer_norm_forward_kernel[grid](
        X, W, B, Y, Mean, RSTD,
        X.stride(0), X.stride(1), W.stride(0), B.stride(0),
        Y.stride(0), Y.stride(1), Mean.stride(0), RSTD.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return Y, Mean, RSTD


def layer_norm_backward(X, W, B, DY, Mean, RSTD):
    n_rows, n_cols = X.shape
    DX = torch.empty_like(X)
    DW = torch.zeros_like(W)
    DB = torch.zeros_like(B)

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    grid = lambda meta: (n_rows,)
    _layer_norm_backward_kernel[grid](
        X, W, B, DY, Mean, RSTD,
        DX, DW, DB,
        n_rows, n_cols,
        X.stride(0), X.stride(1), W.stride(0), B.stride(0),
        DY.stride(0), DY.stride(1), DX.stride(0), DX.stride(1),
        DW.stride(0), DB.stride(0),
        Mean.stride(0), RSTD.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )

    return DX, DW, DB


class LigerLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B):
        Y, Mean, RSTD = layer_norm_forward(X, W, B)
        ctx.save_for_backward(X, W, B, Y, Mean, RSTD)
        return Y

    @staticmethod
    def backward(ctx, dY):
        X, W, B, Y, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = layer_norm_backward(X, W, B, dY, Mean, RSTD)
        return DX, DW, DB
