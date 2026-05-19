import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(meta={"BLOCK_N": 128, "NUM_WARPS": 4}),
        triton.Config(meta={"BLOCK_N": 256, "NUM_WARPS": 8}),
    ],
    key=["n_features"],
)
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    x_ptr, w_ptr, b_ptr, r_ptr, y_ptr,
    mean_ptr, var_ptr,  # for optional storage of mean/var if needed
    n_rows, n_features,
    RESIDUAL: tl.constexpr,  # bool
    BIAS: tl.constexpr,      # bool
    RMS: tl.constexpr,       # bool
    BLOCK_N: tl.constexpr
):
    row_id = tl.program_id(0)
    # each block processes one row
    row_offset = row_id * n_features
    # check bounds
    if row_id >= n_rows:
        return

    # pointers to this row's data
    x_offset = x_ptr + row_offset
    w_offset = w_ptr
    if BIAS:
        b_offset = b_ptr
    if RESIDUAL:
        r_offset = r_ptr + row_offset

    # pgm range
    idx = tl.arange(0, BLOCK_N)
    mask = idx < n_features

    # load data
    x = tl.where(mask, x_offset + idx, 0.0)
    x_val = tl.load(x, mask=mask, other=0.0)
    # compute mean
    mean_val = tl.sum(x_val, axis=0) / n_features
    # compute variance or RMS
    if RMS:
        var_val = tl.sqrt(tl.sum(x_val * x_val, axis=0) / n_features + 1e-5)
    else:
        diff = x_val - mean_val
        var_val = tl.sqrt(tl.sum(diff * diff, axis=0) / n_features + 1e-5)

    # normalize
    if RMS:
        x_norm = x_val / var_val
    else:
        x_norm = (x_val - mean_val) / var_val

    # scale
    w = tl.where(mask, w_offset + idx, 1.0)
    w_val = tl.load(w, mask=mask, other=1.0)
    out = x_norm * w_val

    # bias
    if BIAS:
        b = tl.where(mask, b_offset + idx, 0.0)
        b_val = tl.load(b, mask=mask, other=0.0)
        out = out + b_val

    # residual
    if RESIDUAL:
        r_val = tl.load(tl.where(mask, r_offset + idx, 0.0), mask=mask, other=0.0)
        out = out + r_val

    # store output
    y_out = y_ptr + row_offset
    tl.store(y_out + idx, out, mask=mask)

    # optionally store mean / var for backward
    tl.store(mean_ptr + row_id, mean_val)
    tl.store(var_ptr + row_id, var_val)


@triton.autotune(
    configs=[
        triton.Config(meta={"BLOCK_N": 128, "NUM_WARPS": 4}),
        triton.Config(meta={"BLOCK_N": 256, "NUM_WARPS": 8}),
    ],
    key=["n_features"],
)
@triton.jit
def _layer_norm_bwd_kernel(
    x_ptr, w_ptr, b_ptr, r_ptr, dy_ptr,
    mean_ptr, var_ptr, dx_ptr, dw_ptr, db_ptr,
    n_rows, n_features,
    RESIDUAL: tl.constexpr,  # bool
    BIAS: tl.constexpr,      # bool
    RMS: tl.constexpr,       # bool
    BLOCK_N: tl.constexpr
):
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return

    row_offset = row_id * n_features
    x_offset = x_ptr + row_offset
    w_offset = w_ptr
    if BIAS:
        b_offset = b_ptr
    if RESIDUAL:
        r_offset = r_ptr + row_offset

    mean_val = tl.load(mean_ptr + row_id)
    var_val = tl.load(var_ptr + row_id)

    idx = tl.arange(0, BLOCK_N)
    mask = idx < n_features

    # load x and dy
    x_val = tl.load(tl.where(mask, x_offset + idx, 0), mask=mask, other=0.0)
    dy_val = tl.load(tl.where(mask, dy_ptr + row_offset + idx, 0), mask=mask, other=0.0)
    w_val = tl.load(tl.where(mask, w_offset + idx, 1.0), mask=mask, other=1.0)

    # re-normalize x if RMS is not used
    if RMS:
        x_norm = x_val / var_val
    else:
        x_norm = (x_val - mean_val) / var_val

    # partial grads
    d_out = dy_val
    d_out_w = d_out * x_norm
    dx_norm = d_out * w_val

    # compute sums for grad computation
    sum_dx_norm = tl.sum(dx_norm, axis=0)
    if RMS:
        sum_x_dx_norm = tl.sum(x_val * dx_norm, axis=0)
    else:
        sum_x_dx_norm = tl.sum((x_val - mean_val) * dx_norm, axis=0)

    # backward var or RMS
    if RMS:
        inv = 1.0 / (var_val + 1e-5)
        dx = (dx_norm - x_val * sum_x_dx_norm / (n_features * var_val * var_val)) * inv
    else:
        inv = 1.0 / (n_features * var_val)
        dx = (dx_norm - (sum_dx_norm + (x_val - mean_val) * sum_x_dx_norm * inv)) * inv

    # store dx
    tl.store(dx_ptr + row_offset + idx, tl.where(mask, dx, 0.0), mask=mask)

    # grad w
    block_dw = tl.sum(d_out_w, axis=0)
    if BIAS:
        block_db = tl.sum(d_out, axis=0)
    else:
        block_db = 0.0

    # store partial grads
    tl.atomic_add(dw_ptr + idx, tl.where(mask, d_out_w, 0.0))
    if BIAS:
        tl.atomic_add(db_ptr + idx, tl.where(mask, d_out, 0.0))

    # optional residual gradient accumulation
    if RESIDUAL:
        # if using residual, partial grad for r
        r_val = tl.load(tl.where(mask, r_offset + idx, 0.0), mask=mask, other=0.0)
        dr_val = d_out  # gradient w.r.t. residual is the same as out grad
        tl.atomic_add(dx_ptr + row_offset + idx, tl.where(mask, dr_val, 0.0))


def layer_norm_fwd(
    x, w, b, residual, n_rows, n_features,
    use_residual=False, use_bias=False, use_rms=False
):
    import torch
    y = torch.empty_like(x)
    mean = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    var = torch.empty(n_rows, dtype=x.dtype, device=x.device)

    grid = lambda meta: (n_rows,)
    _layer_norm_fwd_1pass_kernel[grid](
        x, w, b, residual, y,
        mean, var,
        n_rows, n_features,
        RESIDUAL=use_residual,
        BIAS=use_bias,
        RMS=use_rms,
        BLOCK_N=_layer_norm_fwd_1pass_kernel.best_config.kwargs["BLOCK_N"]
    )
    return y, mean, var


def layer_norm_bwd(
    x, w, b, residual, dy,
    mean, var, n_rows, n_features,
    use_residual=False, use_bias=False, use_rms=False
):
    import torch
    dx = torch.empty_like(x)
    dw = torch.zeros_like(w)
    db = torch.zeros_like(b) if b is not None else None

    grid = lambda meta: (n_rows,)
    _layer_norm_bwd_kernel[grid](
        x, w, b, residual, dy,
        mean, var, dx, dw, db if db is not None else w,  # dummy if db is None
        n_rows, n_features,
        RESIDUAL=use_residual,
        BIAS=(b is not None and use_bias),
        RMS=use_rms,
        BLOCK_N=_layer_norm_bwd_kernel.best_config.kwargs["BLOCK_N"]
    )
    return dx, dw, db
