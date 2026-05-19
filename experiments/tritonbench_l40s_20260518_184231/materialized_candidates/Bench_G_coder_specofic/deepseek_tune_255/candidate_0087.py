import torch
import triton
import triton.language as tl
from typing import Optional
from enum import Enum

class ReductionMode(str, Enum):
    NONE = "none"
    SUM = "sum"
    MEAN = "mean"
    BATCHMEAN = "batchmean"

_REDUCTION_MODE_NONE = "none"
_REDUCTION_MODE_SUM = "sum"
_REDUCTION_MODE_MEAN = "mean"
_REDUCTION_MODE_BATCHMEAN = "batchmean"

_str_to_reduction_mode = {
    "none": _REDUCTION_MODE_NONE,
    "sum": _REDUCTION_MODE_SUM,
    "mean": _REDUCTION_MODE_MEAN,
    "batchmean": _REDUCTION_MODE_BATCHMEAN,
}

MAX_FUSED_SIZE = 65536 // 4

def get_num_warps(BLOCK_SIZE):
    if BLOCK_SIZE >= 32768:
        return 32
    if BLOCK_SIZE >= 4096:
        return 16
    if BLOCK_SIZE >= 512:
        return 8
    return 4

@triton.jit
def _kldiv_kernel_forward(
    y_ptr,
    gt_ptr,
    loss_ptr,
    y_batch_stride,
    gt_batch_stride,
    y_last_stride,
    gt_last_stride,
    y_stride_h,
    gt_stride_h,
    y_stride_d,
    gt_stride_d,
    n_elements,
    n_batches,
    n_classes,
    log_target: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    REDUCTION_MODE: tl.constexpr,
):
    BLOCK_SIZE = min(MAX_FUSED_SIZE, BLOCK_SIZE)
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    y_offs = row_idx * y_stride_h + col_offs
    gt_offs = row_idx * gt_stride_h + col_offs
    col_mask = col_offs < n_classes

    y_ptrs = y_ptr + y_offs
    gt_ptrs = gt_ptr + gt_offs

    y = tl.load(y_ptrs, mask=col_mask, other=1e-5)
    gt = tl.load(gt_ptrs, mask=col_mask, other=1e-5)

    if log_target:
        gt = -tl.exp(gt) * gt
    else:
        gt = -gt

    kl = gt * tl.log(y)

    loss = tl.sum(kl, axis=0) / n_batches

    if REDUCTION_MODE == _REDUCTION_MODE_SUM:
        pass
    elif REDUCTION_MODE == _REDUCTION_MODE_MEAN:
        loss /= n_classes
    elif REDUCTION_MODE == _REDUCTION_MODE_BATCHMEAN:
        loss /= n_batches

    loss_ptr += row_idx
    tl.store(loss_ptr, loss)

@triton.jit
def _kldiv_kernel_backward(
    grad_output_ptr,
    y_ptr,
    gt_ptr,
    y_batch_stride,
    gt_batch_stride,
    y_last_stride,
    gt_last_stride,
    y_stride_h,
    gt_stride_h,
    y_stride_d,
    gt_stride_d,
    n_elements,
    n_batches,
    n_classes,
    log_target: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    REDUCTION_MODE: tl.constexpr,
):
    BLOCK_SIZE = min(MAX_FUSED_SIZE, BLOCK_SIZE)
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    y_offs = row_idx * y_stride_h + col_offs
    gt_offs = row_idx * gt_stride_h + col_offs
    col_mask = col_offs < n_classes

    y_ptrs = y_ptr + y_offs
    gt_ptrs = gt_ptr + gt_offs

    y = tl.load(y_ptrs, mask=col_mask, other=1e-5)
    gt = tl.load(gt_ptrs, mask=col_mask, other=1e-5)

    if log_target:
        gt = -tl.exp(gt) * gt
    else:
        gt = -gt

    grad = tl.load(grad_output_ptr + row_idx)

    kl = gt * tl.log(y)
    y = y * grad

    tl.store(y_ptrs, y, mask=col_mask)
    tl.store(gt_ptrs, kl, mask=col_mask)

def kldiv_forward_triton(
    y: torch.Tensor,
    gt: torch.Tensor,
    log_target: bool = False,
    reduction: Optional[ReductionMode] = None,
):
    n_elements = y.numel()
    n_classes = y.size(-1)
    n_batches = gt.size(0)

    if reduction is None:
        reduction = ReductionMode.NONE

    REDUCTION_MODE = _str_to_reduction_mode[reduction]

    loss = torch.empty(n_classes, dtype=torch.float32, device=y.device)

    BLOCK_SIZE = triton.next_power_of_2(n_classes)
    num_warps = get_num_warps(BLOCK_SIZE)

    grid = (n_classes,)

    _kldiv_kernel_forward[grid](
        y,
        gt,
        loss,
        *gt.stride(),
        y.stride(0),
        y.stride(1),
        gt.stride(0),
        gt.stride(1),
        y.stride(0),
        gt.stride(0),
        n_elements,
        n_batches,
        n_classes,
        log_target,
        BLOCK_SIZE,
        REDUCTION_MODE,
        num_warps=num_warps,
        num_stages=1,
    )

    if reduction == ReductionMode.NONE:
        return loss
    elif reduction == ReductionMode.SUM:
        return torch.sum(loss)
    elif reduction == ReductionMode.MEAN:
        return torch.mean(loss)
    elif reduction == ReductionMode.BATCHMEAN:
        return torch.mean(loss[:, None] * n_batches, axis=0)

def kldiv_backward_triton(
    grad_output: torch.Tensor,
    y: torch.Tensor,
    gt: torch.Tensor,
    log_target: bool = False,
):
    assert grad_output.numel() == 1

    n_elements = y.numel()
    n_classes = y.size(-1)

    BLOCK_SIZE = triton.next_power_of_2(n_classes)
    num_warps = get_num_warps(BLOCK_SIZE)

    grid = (n_classes,)

    _kldiv_kernel_backward[grid](
        grad_output,
        y,
        gt,
        *gt.stride(),
        y.stride(0),
        y.stride(1),
        gt.stride(0),
        g
