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
    elif BLOCK_SIZE >= 16384:
        return 16
    elif BLOCK_SIZE >= 8192:
        return 8
    else:
        return 4

@triton.jit
def _kldiv_kernel_forward(y_ptr, gt_ptr, loss_ptr, y_stride, gt_stride, loss_stride, num_cols, reduction_mode, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_cols

    y = tl.load(y_ptr + offsets * y_stride, mask=mask)
    gt = tl.load(gt_ptr + offsets * gt_stride, mask=mask)

    kl = gt * (tl.log(gt) - tl.log(y))
    tl.store(loss_ptr + offsets * loss_stride, kl, mask=mask)

    if reduction_mode == _REDUCTION_MODE_SUM:
        sum_kl = tl.sum(kl, axis=0)
        tl.atomic_add(loss_ptr, sum_kl)
    elif reduction_mode == _REDUCTION_MODE_MEAN:
        mean_kl = tl.sum(kl, axis=0) / num_cols
        tl.atomic_add(loss_ptr, mean_kl)
    elif reduction_mode == _REDUCTION_MODE_BATCHMEAN:
        batch_mean_kl = tl.sum(kl, axis=0) / tl.num_programs()
        tl.atomic_add(loss_ptr, batch_mean_kl)

@triton.jit
def _kldiv_kernel_backward(y_ptr, gt_ptr, grad_ptr, y_stride, gt_stride, grad_stride, num_cols, reduction_mode, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_cols

    y = tl.load(y_ptr + offsets * y_stride, mask=mask)
    gt = tl.load(gt_ptr + offsets * gt_stride, mask=mask)

    grad = -gt / y
    tl.store(grad_ptr + offsets * grad_stride, grad, mask=mask)

    if reduction_mode == _REDUCTION_MODE_SUM:
        sum_grad = tl.sum(grad, axis=0)
        tl.atomic_add(grad_ptr, sum_grad)
    elif reduction_mode == _REDUCTION_MODE_MEAN:
        mean_grad = tl.sum(grad, axis=0) / num_cols
        tl.atomic_add(grad_ptr, mean_grad)
    elif reduction_mode == _REDUCTION_MODE_BATCHMEAN:
        batch_mean_grad = tl.sum(grad, axis=0) / tl.num_programs()
        tl.atomic_add(grad_ptr, batch_mean_grad)

def kldiv_forward_triton(y, gt, reduction='none'):
    reduction_mode = _str_to_reduction_mode[reduction]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, y.shape[-1])
    num_warps = get_num_warps(BLOCK_SIZE)

    loss = torch.empty_like(y)

    grid = lambda meta: (triton.cdiv(y.shape[-1], BLOCK_SIZE),)
    _kldiv_kernel_forward[grid](
        y, gt, loss,
        y.stride(-1), gt.stride(-1), loss.stride(-1),
        y.shape[-1], reduction_mode,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )

    if reduction_mode == _REDUCTION_MODE_SUM:
        return loss.sum()
    elif reduction_mode == _REDUCTION_MODE_MEAN:
        return loss.mean()
    elif reduction_mode == _REDUCTION_MODE_BATCHMEAN:
        return loss.sum() / y.shape[0]
    else:
        return loss

def kldiv_backward_triton(y, gt, grad_output, reduction='none'):
    reduction_mode = _str_to_reduction_mode[reduction]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, y.shape[-1])
    num_warps = get_num_warps(BLOCK_SIZE)

    grad = torch.empty_like(y)

    grid = lambda meta: (triton.cdiv(y.shape[-1], BLOCK_SIZE),)
    _kldiv_kernel_backward[grid](
        y, gt, grad,
        y.stride(-1), gt.stride(-1), grad.stride(-1),
        y.shape[-1], reduction_mode,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )

    if grad_output.ndim == 0 and grad_output.item() == 1:
        return grad
    else:
        return grad * grad_output
