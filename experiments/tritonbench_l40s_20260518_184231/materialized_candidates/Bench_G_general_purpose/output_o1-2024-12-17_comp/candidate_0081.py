import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_fused(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    Mean_ptr, Rstd_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_w, stride_b,
    stride_ym, stride_yn,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x_ptrs = X_ptr + row * stride_xm + offs * stride_xn
    x = tl.where(mask, tl.load(x_ptrs, mask=mask), 0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / N

    # Compute variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # Apply scale and shift
    w_ptrs = W_ptr + offs * stride_w
    b_ptrs = B_ptr + offs * stride_b
    w = tl.where(mask, tl.load(w_ptrs, mask=mask), 0.0)
    b = tl.where(mask, tl.load(b_ptrs, mask=mask), 0.0)

    y = diff * rstd * w + b

    # Store outputs
    y_ptrs = Y_ptr + row * stride_ym + offs * stride_yn
    tl.store(y_ptrs, y, mask=mask)

    # Store mean and rstd for bwd
    if tl.first(offs):
        tl.store(Mean_ptr + row, mean)
        tl.store(Rstd_ptr + row, rstd)


@triton.jit
def _layer_norm_bwd_dx_fused(
    DY_ptr, X_ptr, W_ptr,
    Mean_ptr, Rstd_ptr,
    DX_ptr, DW_partial_ptr, DB_partial_ptr,
    M, N,
    stride_dym, stride_dyn,
    stride_xm, stride_xn,
    stride_w, stride_mean, stride_rstd,
    stride_dxm, stride_dxn,
    STRIDE_PARTIAL_W, STRIDE_PARTIAL_B,
    LOCK_PTR,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    dy_ptrs = DY_ptr + row * stride_dym + offs * stride_dyn
    x_ptrs = X_ptr + row * stride_xm + offs * stride_xn
    w_ptrs = W_ptr + offs * stride_w

    dy = tl.where(mask, tl.load(dy_ptrs, mask=mask), 0.0)
    x = tl.where(mask, tl.load(x_ptrs, mask=mask), 0.0)
    w = tl.where(mask, tl.load(w_ptrs, mask=mask), 0.0)

    mean = tl.load(Mean_ptr + row * stride_mean)
    rstd = tl.load(Rstd_ptr + row * stride_rstd)

    diff = x - mean
    # normalized gradient
    dy_norm = dy * w

    # partial sums for dW and dB
    partial_dw = dy * diff * rstd
    partial_db = dy

    sum_dy_norm = tl.sum(dy_norm, axis=0)
    sum_dy_norm_x = tl.sum(dy_norm * diff, axis=0)

    dx = (dy_norm - (sum_dy_norm / N) - (diff * rstd * sum_dy_norm_x / N)) * rstd

    dx_ptrs = DX_ptr + row * stride_dxm + offs * stride_dxn
    tl.store(dx_ptrs, dx, mask=mask)

    # Accumulate partial dW, dB using locks
    lock_id = offs // BLOCK_SIZE
    lock_ptrs = LOCK_PTR + lock_id
    # Acquire lock
    acquired = 0
    while acquired == 0:
        acquired = tl.atomic_cas(lock_ptrs, 0, 1)

    dw_ptrs = DW_partial_ptr + pid * STRIDE_PARTIAL_W + offs
    db_ptrs = DB_partial_ptr + pid * STRIDE_PARTIAL_B + offs
    tl.store(dw_ptrs, partial_dw, mask=mask)
    tl.store(db_ptrs, partial_db, mask=mask)

    # Release lock
    tl.atomic_xchg(lock_ptrs, 0)


@triton.jit
def _layer_norm_bwd_dwdb(
    DW_partial_ptr, DB_partial_ptr,
    FINAL_DW_ptr, FINAL_DB_ptr,
    M, N,
    STRIDE_PARTIAL_W, STRIDE_PARTIAL_B,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # sum over M
    dw_sum = 0.0
    db_sum = 0.0
    # Combine partial results from M rows
    for m in range(M):
        dw_ptrs = DW_partial_ptr + m * STRIDE_PARTIAL_W + offs
        db_ptrs = DB_partial_ptr + m * STRIDE_PARTIAL_B + offs
        dw_val = tl.where(mask, tl.load(dw_ptrs, mask=mask), 0.0)
        db_val = tl.where(mask, tl.load(db_ptrs, mask=mask), 0.0)
        dw_sum += tl.sum(dw_val, axis=0)
        db_sum += tl.sum(db_val, axis=0)

    # store final
    if tl.first(offs):
        FINAL_DW_ptr[pid] = dw_sum
        FINAL_DB_ptr[pid] = db_sum


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        # shapes
        M, N = x.shape
        x_ = x.contiguous()
        w_ = w.contiguous()
        b_ = b.contiguous()

        y = torch.empty_like(x_)
        mean = torch.empty([M], dtype=x.dtype, device=x.device)
        rstd = torch.empty([M], dtype=x.dtype, device=x.device)

        BLOCK_SIZE = 128
        grid = lambda META: (M,)

        _layer_norm_fwd_fused[grid](
            x_, w_, b_, y,
            mean, rstd,
            M, N,
            x_.stride(0), x_.stride(1),
            w_.stride(0), b_.stride(0),
            y.stride(0), y.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(x_, w_, b_, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x_, w_, b_, mean, rstd = ctx.saved_tensors
        M, N = x_.shape

        dx = torch.empty_like(x_)
        dw = torch.empty_like(w_)
        db = torch.empty_like(b_)

        # partial buffers
        dw_partial = torch.empty([M, N], dtype=x_.dtype, device=x_.device)
        db_partial = torch.empty([M, N], dtype=x_.dtype, device=x_.device)
        lock = torch.zeros([N // 128 + 1], dtype=torch.int32, device=x_.device)

        BLOCK_SIZE = 128
        grid = lambda META: (M,)

        _layer_norm_bwd_dx_fused[grid](
            dy, x_, w_,
            mean, rstd,
            dx, dw_partial, db_partial,
            M, N,
            dy.stride(0), dy.stride(1),
            x_.stride(0), x_.stride(1),
            w_.stride(0), mean.stride(0), rstd.stride(0),
            dx.stride(0), dx.stride(1),
            dw_partial.stride(0), db_partial.stride(0),
            lock,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # finalize dw, db
        _layer_norm_bwd_dwdb[(N // BLOCK_SIZE,)](
            dw_partial, db_partial,
            dw, db,
            M, N,
            dw_partial.stride(0), db_partial.stride(0),
            BLOCK_SIZE=BLOCK_SIZE
        )

        return dx, dw, db


class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias)
