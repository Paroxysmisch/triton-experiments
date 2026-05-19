import torch
import triton
import triton.language as tl

# Constants
MAX_FUSED_SIZE = 65536 // 4
_REDUCTION_MODE_NONE = 0
_REDUCTION_MODE_SUM = 1
_REDUCTION_MODE_MEAN = 2
_REDUCTION_MODE_BATCHMEAN = 3

_str_to_reduction_mode = {
    'none': _REDUCTION_MODE_NONE,
    'sum': _REDUCTION_MODE_SUM,
    'mean': _REDUCTION_MODE_MEAN,
    'batchmean': _REDUCTION_MODE_BATCHMEAN
}

def get_num_warps(BLOCK_SIZE):
    if BLOCK_SIZE >= 32768:
        return 32
    if BLOCK_SIZE >= 16384:
        return 16
    if BLOCK_SIZE >= 8192:
        return 8
    return 4

@triton.jit
def _kldiv_kernel_forward(
    y_ptr, gt_ptr, loss_ptr,
    y_stride_0, y_stride_1,
    gt_stride_0, gt_stride_1,
    loss_stride_0, loss_stride_1,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    REDUCTION_MODE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y_row_ptr = y_ptr + row_idx * y_stride_0
    gt_row_ptr = gt_ptr + row_idx * gt_stride_0
    loss_row_ptr = loss_ptr + row_idx * loss_stride_0

    y = tl.load(y_row_ptr + col_offsets * y_stride_1, mask=mask)
    gt = tl.load(gt_row_ptr + col_offsets * gt_stride_1, mask=mask)

    kl = gt * (tl.log(gt) - tl.log(y))
    kl = tl.where(mask, kl, 0.0)

    if REDUCTION_MODE == _REDUCTION_MODE_NONE:
        tl.store(loss_row_ptr + col_offsets * loss_stride_1, kl, mask=mask)
    elif REDUCTION_MODE == _REDUCTION_MODE_SUM:
        loss = tl.sum(kl)
        if col_block_idx == 0:
            tl.store(loss_row_ptr, loss)
    elif REDUCTION_MODE in (_REDUCTION_MODE_MEAN, _REDUCTION_MODE_BATCHMEAN):
        loss = tl.sum(kl) / n_cols
        if col_block_idx == 0:
            tl.store(loss_row_ptr, loss)

@triton.jit
def _kldiv_kernel_backward(
    y_ptr, gt_ptr, grad_ptr,
    y_stride_0, y_stride_1,
    gt_stride_0, gt_stride_1,
    grad_stride_0, grad_stride_1,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    LOG_TARGET: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y_row_ptr = y_ptr + row_idx * y_stride_0
    gt_row_ptr = gt_ptr + row_idx * gt_stride_0
    grad_row_ptr = grad_ptr + row_idx * grad_stride_0

    y = tl.load(y_row_ptr + col_offsets * y_stride_1, mask=mask)
    gt = tl.load(gt_row_ptr + col_offsets * gt_stride_1, mask=mask)

    if LOG_TARGET:
        grad = -tl.exp(gt) * gt
    else:
        grad = -gt

    grad = tl.where(mask, grad, 0.0)
    tl.store(grad_row_ptr + col_offsets * grad_stride_1, grad, mask=mask)

def kldiv_forward_triton(y, gt, reduction='mean'):
    assert y.is_contiguous()
    assert gt.is_contiguous()
    assert y.shape == gt.shape
    assert y.dtype == gt.dtype
    assert reduction in _str_to_reduction_mode

    REDUCTION_MODE = _str_to_reduction_mode[reduction]
    n_rows, n_cols = y.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    num_warps = get_num_warps(BLOCK_SIZE)

    if REDUCTION_MODE == _REDUCTION_MODE_NONE:
        loss = torch.empty_like(y)
    else:
        loss = torch.empty(n_rows, dtype=y.dtype, device=y.device)

    grid = (n_rows, triton.cdiv(n_cols, BLOCK_SIZE))

    _kldiv_kernel_forward[grid](
        y, gt, loss,
        y.stride(0), y.stride(1),
        gt.stride(0), gt.stride(1),
        loss.stride(0), 1 if REDUCTION_MODE == _REDUCTION_MODE_NONE else 0,
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        REDUCTION_MODE=REDUCTION_MODE,
        num_warps=num_warps,
    )

    if REDUCTION_MODE == _REDUCTION_MODE_MEAN:
        return loss.mean()
    elif REDUCTION_MODE == _REDUCTION_MODE_SUM:
        return loss.sum()
    elif REDUCTION_MODE == _REDUCTION_MODE_BATCHMEAN:
        return loss.sum() / n_rows
    else:
        return loss

def kldiv_backward_triton(y, gt, grad_output, log_target=False):
    assert y.is_contiguous()
    assert gt.is_contiguous()
    assert y.shape == gt.shape
    assert y.dtype == gt.dtype

    n_rows, n_cols = y.shape
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    num_warps = get_num_warps(BLOCK_SIZE)

    grad = torch.empty_like(y)
    grid = (n_rows, triton.cdiv(n_cols, BLOCK_SIZE))

    _kldiv_kernel_backward[grid](
        y, gt, grad,
        y.stride(0), y.stride(1),
        gt.stride(0), gt.stride(1),
        grad.stride(0), grad.stride(1),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        LOG_TARGET=log_target,
        num_warps=num_warps,
    )

    if grad_output.numel() == 1 and grad_output.item() == 1:
        return grad
    else:
        return grad * grad_output.view(-1, 1)
