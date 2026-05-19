import triton
import triton.language as tl
import torch

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
    elif BLOCK_SIZE >= 8192:
        return 16
    elif BLOCK_SIZE >= 2048:
        return 8
    else:
        return 4

@triton.jit
def _kldiv_kernel_forward(
    y_ptr, y_row_stride,
    gt_ptr, gt_row_stride,
    loss_ptr, loss_row_stride,
    n_rows, n_cols,
    eps,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y_row = y_ptr + row_idx * y_row_stride
    gt_row = gt_ptr + row_idx * gt_row_stride
    loss_row = loss_ptr + row_idx * loss_row_stride

    y = tl.load(y_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    gt = tl.load(gt_row + col_offsets, mask=mask, other=0.0).to(tl.float32)

    log_gt = tl.log(gt + eps)
    log_y = tl.log(y + eps)
    term = gt * (log_gt - log_y)

    if reduction == _REDUCTION_MODE_NONE:
        tl.store(loss_row + col_offsets, term, mask=mask)
    else:
        row_sum = tl.sum(term)
        tl.atomic_add(loss_ptr, row_sum)

def kldiv_forward_triton(y, gt, reduction='mean', eps=1e-8):
    assert reduction in _str_to_reduction_mode, f"Invalid reduction mode: {reduction}"
    reduction_mode = _str_to_reduction_mode[reduction]

    original_shape = y.shape
    y = y.view(-1, y.size(-1))
    gt = gt.view(-1, gt.size(-1))
    n_rows, n_cols = y.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Block size {BLOCK_SIZE} exceeds maximum fused size {MAX_FUSED_SIZE}")
    num_warps = get_num_warps(BLOCK_SIZE)

    if reduction == 'none':
        loss = torch.empty_like(y)
    else:
        loss = torch.zeros(1, device=y.device, dtype=torch.float32)

    _kldiv_kernel_forward[(n_rows,)](
        y_ptr=y,
        y_row_stride=y.stride(0),
        gt_ptr=gt,
        gt_row_stride=gt.stride(0),
        loss_ptr=loss,
        loss_row_stride=loss.stride(0) if reduction == 'none' else 0,
        n_rows=n_rows,
        n_cols=n_cols,
        eps=eps,
        reduction=reduction_mode,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    if reduction == 'mean':
        loss = loss / (n_rows * n_cols)
    elif reduction == 'batchmean':
        loss = loss / n_rows
    return loss.view(original_shape) if reduction == 'none' else loss

@triton.jit
def _kldiv_kernel_backward(
    dY_ptr, dY_row_stride,
    y_ptr, y_row_stride,
    gt_ptr, gt_row_stride,
    dloss_ptr, dloss_row_stride,
    n_rows, n_cols,
    eps,
    reduction: tl.constexpr,
    log_target: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y_row = y_ptr + row_idx * y_row_stride
    gt_row = gt_ptr + row_idx * gt_row_stride
    dloss_row = dloss_ptr + row_idx * dloss_row_stride

    y = tl.load(y_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    gt = tl.load(gt_row + col_offsets, mask=mask, other=0.0).to(tl.float32)

    if log_target:
        gt_vals = tl.exp(gt)
    else:
        gt_vals = gt

    grad_elem = -gt_vals / (y + eps)

    if reduction == _REDUCTION_MODE_NONE:
        dY = tl.load(dY_ptr + row_idx * dY_row_stride + col_offsets, mask=mask, other=0.0).to(tl.float32)
    else:
        dY = tl.load(dY_ptr).to(tl.float32)

    grad_elem *= dY
    tl.store(dloss_row + col_offsets, grad_elem, mask=mask)

def kldiv_backward_triton(grad_output, y, gt, reduction='mean', log_target=False, eps=1e-8):
    reduction_mode = _str_to_reduction_mode[reduction]

    original_shape = y.shape
    y = y.view(-1, y.size(-1))
    gt = gt.view(-1, gt.size(-1))
    n_rows, n_cols = y.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Block size {BLOCK_SIZE} exceeds maximum fused size {MAX_FUSED_SIZE}")
    num_warps = get_num_warps(BLOCK_SIZE)

    dloss = torch.empty_like(y)

    if reduction != 'none' and grad_output.numel() != 1:
        raise ValueError("grad_output must be a scalar tensor when reduction is not 'none'")

    _kldiv_kernel_backward[(n_rows,)](
        dY_ptr=grad_output if reduction == 'none' else grad_output.data_ptr(),
        dY_row_stride=grad_output.stride(0) if reduction == 'none' else 0,
        y_ptr=y,
        y_row_stride=y.stride(0),
        gt_ptr=gt,
        gt_row_stride=gt.stride(0),
        dloss_ptr=dloss,
        dloss_row_stride=dloss.stride(0),
        n_rows=n_rows,
        n_cols=n_cols,
        eps=eps,
        reduction=reduction_mode,
        log_target=log_target,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    return dloss.view(original_shape)

class KLDivLossTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, gt, reduction, log_target, eps):
        ctx.reduction = reduction
        ctx.log_target = log_target
        ctx.eps = eps
        ctx.save_for_backward(y, gt)
        loss = kldiv_forward_triton(y, gt, reduction=reduction, eps=eps)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y, gt = ctx.saved_tokens
        dloss = kldiv_backward_triton(grad_output, y, gt, ctx.reduction, ctx.log_target, ctx.eps)
        return dloss, None, None, None, None
