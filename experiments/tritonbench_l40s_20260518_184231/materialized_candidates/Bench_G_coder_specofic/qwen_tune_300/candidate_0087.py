import torch
import triton
import triton.language as tl

# Triton kernel for forward KL divergence calculation
@triton.jit
def _kldiv_kernel_forward(
    y_ptr, gt_ptr, loss_ptr,
    stride_yb, stride_yt,
    stride_gtb, stride_gtnt,
    stride_lossb, stride_lossn,
    n_samples, n_targets: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute block indices
    block_idx = tl.program_id(0)
    y_block_ptr = y_ptr + block_idx * stride_yb
    gt_block_ptr = gt_ptr + block_idx * stride_gtb
    loss_block_ptr = loss_ptr + block_idx * stride_lossb

    # Iterate over target columns
    for i in range(0, n_targets, BLOCK_SIZE):
        col_idx = i + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_targets

        y_ptr_mask = y_block_ptr + (col_idx) * stride_yt
        gt_ptr_mask = gt_block_ptr + (col_idx) * stride_gtnt
        loss_ptr_mask = loss_block_ptr + (col_idx)

        y = tl.load(y_ptr_mask, mask=mask, other=0.0)
        gt = tl.load(gt_ptr_mask, mask=mask, other=0.0)
        gt_log = tl.log(gt)

        # Compute KL divergence
        loss = tl.where(mask, gt * (tl.log(gt) - tl.log(y)), 0.0)

        # Sum or mean reduction
        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr_mask, loss, mask=mask)
        elif reduction == _REDUCTION_MODE_BATCHMEAN:
            tl.store(loss_ptr_mask, loss, mask=mask)
        elif reduction == _REDUCTION_MODE_MEAN:
            loss = tl.sum(loss) / n_samples
            tl.store(loss_ptr_mask, loss, mask=mask)
        elif reduction == _REDUCTION_MODE_SUM:
            loss = tl.sum(loss)
            tl.store(loss_ptr_mask, loss, mask=mask)

# Triton kernel for backward KL divergence calculation
@triton.jit
def _kldiv_kernel_backward(
    gt_ptr, y_ptr, grad_ptr,
    stride_gtb, stride_gtnt,
    stride_yb, stride_yt,
    stride_gradb, stride_gradt,
    n_samples, n_targets: tl.constexpr,
    reduction: tl.constexpr,
    target_is_log: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute block indices
    block_idx = tl.program_id(0)
    gt_block_ptr = gt_ptr + block_idx * stride_gtb
    y_block_ptr = y_ptr + block_idx * stride_yb

    grad_block_ptr = grad_ptr + block_idx * stride_gradb

    # Iterate over target columns
    for i in range(0, n_targets, BLOCK_SIZE):
        col_idx = i + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_targets

        gt_ptr_mask = gt_block_ptr + (col_idx) * stride_gtnt
        y_ptr_mask = y_block_ptr + (col_idx) * stride_yt
        grad_ptr_mask = grad_block_ptr + (col_idx) * stride_gradt

        gt = tl.load(gt_ptr_mask, mask=mask, other=0.0)
        y = tl.load(y_ptr_mask, mask=mask, other=0.0)

        if target_is_log:
            gt = tl.exp(gt)
            gt = tl.where(mask, -gt, 0.0)
        else:
            gt = tl.where(mask, -gt, 0.0)

        # Store gradient
        grad = gt * (tl.log(gt) - tl.log(y))
        tl.store(grad_ptr_mask, grad, mask=mask)

# Function to invoke forward Triton kernel
def kldiv_forward_triton(
    y: torch.Tensor, gt: torch.Tensor,
    reduction: str = "none",
    log_target: bool = False,
) -> torch.Tensor:
    assert reduction in _str_to_reduction_mode

    n_samples, n_targets = gt.shape
    assert y.shape == (n_samples, n_targets)

    BLOCK_SIZE = min(65536 // 4, max_fused_size)
    n_warps = get_num_warps(BLOCK_SIZE)

    if log_target:
        gt = gt.exp()

    loss = torch.empty(
        n_samples if reduction == "batchmean" else n_samples, n_targets,
        device=gt.device, dtype=gt.dtype,
    )

    _kldiv_kernel_forward[(n_samples,)](
        y, gt, loss,
        y.stride(0), y.stride(1),
        gt.stride(0), gt.stride(1),
        loss.stride(0), loss.stride(1),
        n_samples, n_targets,
        _str_to_reduction_mode[reduction],
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=n_warps,
    )

    if reduction == "batchmean":
        loss = loss.sum() / n_samples
    elif reduction != "none":
        loss = loss.mean(dim=0)

    return loss

# Function to invoke backward Triton kernel
def kldiv_backward_triton(
    gt: torch.Tensor, y: torch.Tensor, grad_output: torch.Tensor,
    log_target: bool = False,
) -> torch.Tensor:
    assert grad_output.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert grad_output.device == y.device

    n_samples, n_targets = gt.shape
    assert y.shape == (n_samples, n_targets)

    BLOCK_SIZE = min(65536 // 4, max_fused_size)
    n_warps = get_num_warps(BLOCK_SIZE)

    if log_target:
        gt = gt.exp()

    grad = torch.empty_like(y, dtype=y.dtype)

    _kldiv_kernel_backward[(n_samples,)](
        gt, y, grad,
        gt.stride(0), gt.stride(1),
        y.stride(0), y.stride(1),
        grad.stride(0), grad.stride(1),
        n_samples, n_targets,
        reduction=_REDUCTION_MODE_NONE,
        target_is_log=log_target,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=n_warps,
    )

    if grad_output.numel() == 1:
        return grad
    else:
        assert grad_output.shape == grad.shape
        return (grad_output[:, None] * grad).sum() / n_samples
