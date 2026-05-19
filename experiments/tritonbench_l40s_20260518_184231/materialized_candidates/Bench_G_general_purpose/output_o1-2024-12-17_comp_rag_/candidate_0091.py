import triton
import triton.language as tl
import torch

# -------------------------------------------------------------------------------------
# Constants and Utilities
# -------------------------------------------------------------------------------------
MAX_FUSED_SIZE = 65536 // 4

_REDUTION_MODE_NONE = 0
_REDUTION_MODE_SUM = 1
_REDUTION_MODE_MEAN = 2
_REDUTION_MODE_BATCHMEAN = 3

_str_to_reduction_mode = {
    'none': _REDUTION_MODE_NONE,
    'sum': _REDUTION_MODE_SUM,
    'mean': _REDUTION_MODE_MEAN,
    'batchmean': _REDUTION_MODE_BATCHMEAN,
}

next_power_of_2 = triton.next_power_of_2

def get_num_warps(BLOCK_SIZE):
    if BLOCK_SIZE >= 32768:
        return 32
    elif BLOCK_SIZE >= 8192:
        return 16
    elif BLOCK_SIZE >= 2048:
        return 8
    else:
        return 4

def calculate_settings(n):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps = get_num_warps(BLOCK_SIZE)
    return BLOCK_SIZE, num_warps

# -------------------------------------------------------------------------------------
# KLDivergence Triton Kernels
# -------------------------------------------------------------------------------------
@triton.jit
def _kldiv_kernel_forward(
    y_ptr, y_stride,
    gt_ptr, gt_stride,
    loss_ptr, loss_stride,
    n_cols, reduction_mode, log_target,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    col_range = tl.arange(0, BLOCK_SIZE)
    mask = col_range < n_cols

    y_row_ptr = y_ptr + row_id * y_stride
    gt_row_ptr = gt_ptr + row_id * gt_stride
    loss_row_ptr = loss_ptr + row_id * loss_stride

    y_val = tl.load(y_row_ptr + col_range, mask=mask, other=0.0)
    gt_val = tl.load(gt_row_ptr + col_range, mask=mask, other=0.0)

    # KL(y_true || y_pred) = y_true * (log(y_true) - log(y_pred))
    if log_target == 0:
        kl_val = gt_val * (tl.log(gt_val + 1e-12) - tl.log(y_val + 1e-12))
    else:
        # log_target == 1 means gt_val is in log-space
        # KL( exp(gt_val) || y_val ) = exp(gt_val) * (gt_val - log(y_val))
        kl_val = tl.exp(gt_val) * (gt_val - tl.log(y_val + 1e-12))

    if reduction_mode == _REDUTION_MODE_NONE:
        tl.store(loss_row_ptr + col_range, kl_val, mask=mask)
    elif reduction_mode in (_REDUTION_MODE_SUM, _REDUTION_MODE_MEAN, _REDUTION_MODE_BATCHMEAN):
        out_val = tl.sum(kl_val, axis=0)
        if reduction_mode == _REDUTION_MODE_MEAN:
            out_val = out_val / n_cols
        # For batchmean, we handle scaling outside
        if tl.thread_id_x() == 0:
            tl.store(loss_ptr + row_id, out_val)

@triton.jit
def _kldiv_kernel_backward(
    grad_ptr, grad_stride,
    y_ptr, y_stride,
    gt_ptr, gt_stride,
    grad_out_ptr, grad_out_stride,
    n_cols, reduction_mode, log_target,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    col_range = tl.arange(0, BLOCK_SIZE)
    mask = col_range < n_cols

    grad_row_ptr = grad_ptr + row_id * grad_stride
    y_row_ptr = y_ptr + row_id * y_stride
    gt_row_ptr = gt_ptr + row_id * gt_stride
    grad_out_row_ptr = grad_out_ptr + row_id * grad_out_stride

    grad_val = tl.load(grad_row_ptr + col_range, mask=mask, other=0.0)
    y_val = tl.load(y_row_ptr + col_range, mask=mask, other=0.0)
    gt_val = tl.load(gt_row_ptr + col_range, mask=mask, other=0.0)

    if reduction_mode in (_REDUTION_MODE_SUM, _REDUTION_MODE_MEAN, _REDUTION_MODE_BATCHMEAN):
        # broadcast grad_val from single scalar row
        grad_val = tl.broadcast_to(grad_val, [BLOCK_SIZE])

    # d/dy of KL(y_true || y_pred) = - (y_true / y_pred)
    if log_target == 0:
        grad_out = - gt_val / (y_val + 1e-12)
    else:
        # if log target, grad = - exp(gt_val) / y_val
        grad_out = - tl.exp(gt_val) / (y_val + 1e-12)

    if reduction_mode == _REDUTION_MODE_MEAN:
        grad_out = grad_out / n_cols
    elif reduction_mode == _REDUTION_MODE_BATCHMEAN:
        # batchmean needs normalization by total number of elements across the batch
        # We do partial sum across columns, final scaling done outside for total batch size
        pass

    grad_out = grad_out * grad_val
    tl.store(grad_out_row_ptr + col_range, grad_out, mask=mask)

def kldiv_forward_triton(y, gt, reduction='mean', log_target=False):
    reduction_mode = _str_to_reduction_mode[reduction]
    n_rows, n_cols = y.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    if reduction_mode == _REDUTION_MODE_NONE:
        loss = torch.empty_like(y)
    elif reduction_mode in (_REDUTION_MODE_SUM, _REDUTION_MODE_MEAN, _REDUTION_MODE_BATCHMEAN):
        loss = torch.empty((n_rows,), dtype=y.dtype, device=y.device)

    _kldiv_kernel_forward[(n_rows,)](
        y, y.stride(0),
        gt, gt.stride(0),
        loss, loss.stride(0),
        n_cols,
        reduction_mode,
        int(log_target),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    if reduction_mode == _REDUTION_MODE_NONE:
        return loss
    elif reduction_mode == _REDUTION_MODE_SUM:
        return loss.sum()
    elif reduction_mode == _REDUTION_MODE_MEAN:
        return loss.sum() / n_rows
    elif reduction_mode == _REDUTION_MODE_BATCHMEAN:
        return loss.sum() / (n_rows * n_cols)

def kldiv_backward_triton(grad_output, y, gt, reduction='mean', log_target=False):
    reduction_mode = _str_to_reduction_mode[reduction]
    n_rows, n_cols = y.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    grad = torch.empty_like(y)
    if reduction_mode == _REDUTION_MODE_NONE:
        # elementwise grad
        pass
    elif reduction_mode in (_REDUTION_MODE_SUM, _REDUTION_MODE_MEAN, _REDUTION_MODE_BATCHMEAN):
        if grad_output.numel() == 1:
            grad_val = torch.full((n_rows, 1), grad_output.item(), dtype=y.dtype, device=y.device)
        else:
            grad_val = grad_output
        grad = grad_val

    _kldiv_kernel_backward[(n_rows,)](
        grad, grad.stride(0),
        y, y.stride(0),
        gt, gt.stride(0),
        grad, grad.stride(0),
        n_cols,
        reduction_mode,
        int(log_target),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    if reduction_mode == _REDUTION_MODE_BATCHMEAN:
        # final scaling by (1 / (n_rows * n_cols))
        return grad * (1.0 / (n_rows * n_cols))
    return grad

# -------------------------------------------------------------------------------------
# RMS Layernorm Triton Kernels (Document 1)
# -------------------------------------------------------------------------------------
@triton.jit
def _rms_layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    n_cols, eps,
    BLOCK_SIZE : tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0)
    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    inv_var = 1 / tl.math.sqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask = mask)

@triton.jit
def _rms_layernorm_backward(
    dY, dY_row_stride,
    X,   X_row_stride,
    W,   W_row_stride,
    r,   r_row_stride,
    dW, dW_row_stride,
    n_cols, eps,
    BLOCK_SIZE : tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X  += row_idx *  X_row_stride
    r  += row_idx *  r_row_stride

    dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
    X_row  = tl.load(X  + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row  = tl.load(W  + col_offsets, mask = mask, other = 0).to(tl.float32)

    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var
    dY_W = dY_row * W_row
    rowsum_dY_normed = tl.sum(dY_W * normed, axis = 0)
    output = inv_var/n_cols * (n_cols*dY_W - normed*rowsum_dY_normed)
    tl.store(dY + col_offsets, output, mask = mask)

class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, eps):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = "cuda")
        r = torch.empty(n_rows, dtype = torch.float32, device = "cuda")

        _rms_layernorm_forward[(n_rows,)](
            Y, Y.stride(0),
            X, X.stride(0),
            W, W.stride(0),
            r, r.stride(0),
            n_cols, eps,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows, n_cols = dY.shape
        dW = X

        _rms_layernorm_backward[(n_rows,)](
            dY, dY.stride(0),
            X,  X .stride(0),
            W,  W .stride(0),
            r,  r .stride(0),
            dW, dW.stride(0),
            n_cols, ctx.eps,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        dX = dY.view(*shape)
        return dX, None, None, None
