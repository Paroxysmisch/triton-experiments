import torch
import triton
import triton.language as tl
import mamba_ssm.utils as utils
from packaging import version

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, logits_row_stride,
    labels_ptr,
    loss_ptr, loss_row_stride,
    lse_ptr, lse_row_stride,
    n_cols, n_ignore_classes,
    logit_scale,
    smooth_eps,
    curr_class_idx, nproc_per_node,
    IS_DDP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current row index
    row_idx = tl.program_id(0)
    logits_ptr = logits_ptr + row_idx * logits_row_stride
    labels_ptr = labels_ptr + row_idx * n_cols
    loss_ptr = loss_ptr + row_idx * loss_row_stride
    lse_ptr = lse_ptr + row_idx * lse_row_stride

    # Load the label for the current row
    label_idx = tl.load(labels_ptr + n_ignore_classes)

    # If DDP, adjust label index and ignore specific classes
    if IS_DDP:
        label_idx -= curr_class_idx
        ignore_class_idxes = (label_idx < 0) | (label_idx >= n_cols - n_ignore_classes)
        label_idx = tl.where(ignore_class_idxes, -100, label_idx)

    # Compute probabilities for each column
    logits = tl.load(logits_ptr + tl.arange(0, BLOCK_SIZE))
    logits = logits / logit_scale
    max_logits = tl.max(logits, 0)
    logits -= max_logits
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, 0) + 1e-6

    # Compute the loss
    if smooth_eps > 0:
        loss = -tl.log(sum_exp_logits / n_cols)
    else:
        label_logits = tl.load(logits_ptr + label_idx)
        loss = -label_logits
    loss += max_logits
    if smooth_eps > 0:
        loss += smooth_eps * tl.log(sum_exp_logits)

    # Store the loss and log-sum-exp value
    tl.store(loss_ptr, loss)
    tl.store(lse_ptr, max_logits + tl.log(sum_exp_logits))

def cross_entropy_fwd_wrapper(
    logits, labels, n_ignore_classes,
    logit_scale, smooth_eps,
    curr_class_idx, nproc_per_node,
    IS_DDP, n_cols, n_rows,
):
    # Create output tensors
    loss = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
    lse = torch.empty(n_rows, device=logits.device, dtype=torch.float32)

    # Define block size and grid
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows, )

    # Call the Triton kernel
    cross_entropy_fwd_kernel[grid](
        logits, logits.stride(0),
        labels, labels.stride(0),
        loss, loss.stride(0),
        lse, lse.stride(0),
        n_cols, n_ignore_classes,
        logit_scale,
        smooth_eps,
        curr_class_idx, nproc_per_node,
        IS_DDP,
        BLOCK_SIZE,
        num_warps=1,
    )

    return loss, lse

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, logits_row_stride,
    labels_ptr,
    lse_global_ptr,
    dloss_ptr, dloss_row_stride,
    logits_grad_ptr, logits_grad_row_stride,
    n_cols, n_ignore_classes,
    logit_scale,
    smooth_eps,
    curr_class_idx, nproc_per_node,
    IS_DDP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current row index
    row_idx = tl.program_id(0)
    logits_ptr = logits_ptr + row_idx * logits_row_stride
    labels_ptr = labels_ptr + row_idx * n_cols
    dloss_ptr = dloss_ptr + row_idx * dloss_row_stride
    logits_grad_ptr = logits_grad_ptr + row_idx * logits_grad_row_stride

    # Load the label for the current row and the global log-sum-exp value
    label_idx = tl.load(labels_ptr + n_ignore_classes)
    lse_global = tl.load(lse_global_ptr)

    # If DDP, adjust label index and ignore specific classes
    if IS_DDP:
        label_idx -= curr_class_idx
        ignore_class_idxes = (label_idx < 0) | (label_idx >= n_cols - n_ignore_classes)
        label_idx = tl.where(ignore_class_idxes, -100, label_idx)

    # Load the loss gradient
    dloss = tl.load(dloss_ptr)

    # Compute the contribution for the label logits
    logits = tl.load(logits_ptr + label_idx)
    logits_grad = dloss * (tl.exp(logits / logit_scale) / (tl.exp(lse_global) + 1e-6))
    if smooth_eps > 0:
        logits_grad += smooth_eps * (lse_global - logits) / n_cols
    logits_grad -= dloss
    logits_grad *= logit_scale
    tl.store(logits_grad_ptr + label_idx, logits_grad)

    # Compute the contribution for the other logits
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=0.0)
        logits -= lse_global
        logits_grad = tl.exp(logits) * dloss
        if smooth_eps > 0:
            logits_grad += smooth_eps * (tl.where(col_offsets < n_cols, logits, -1e6) - lse_global) / n_cols
        logits_grad *= logit_scale
        tl.store(logits_grad_ptr + col_offsets, logits_grad, mask=col_offsets < n_cols)

def cross_entropy_bwd_wrapper(
    logits, labels, lse_global,
    dloss,
    logits_grad,
    n_ignore_classes,
    logit_scale, smooth_eps,
    curr_class_idx, nproc_per_node,
    IS_DDP, n_cols, n_rows,
    inplace_backward,
):
    # Define block size and grid
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows, )

    # Call the Triton kernel
    if inplace_backward:
        cross_entropy_bwd_kernel[grid](
            logits, logits.stride(0),
            labels, labels.stride(0),
            dloss, dloss.stride(0),
            logits, logits.stride(0),
            n_cols, n_ignore_classes,
            logit_scale,
            smooth_eps,
            curr_class_idx, nproc_per_node,
            IS_DDP,
            BLOCK_SIZE,
            num_warps=1,
        )
    else:
        cross_entropy_bwd_kernel[grid](
            logits, logits.stride(0),
            labels, labels.stride(0),
            dloss, dloss.stride(0),
            logits_grad, logits_grad.stride(0),
            n_cols, n_ignore_classes,
            logit_scale,
            smooth_eps,
            curr_class_idx, nproc_per_node,
            IS_DDP,
            BLOCK_SIZE,
            num_warps=1,
        )

class CrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels, n_ignore_classes,
                logit_scale, smooth_eps,
                curr_class_idx, nproc_per_node,
                is_ddp, ignore_index=-100, is_train=True):
        n_cols, n_rows = logits.shape

        # Adjust labels for the current process
        if is_ddp:
            curr_class_range = torch.arange(curr_class_idx, curr_class_idx + n_cols, device=logits.device)
            labels = torch.where(labels == ignore_index, -100, torch.where(curr_class_range == labels, labels - curr_class_idx, -100))

        # Compute the forward pass
        loss, lse = cross_entropy_fwd_wrapper(
            logits, labels, n_ignore_classes,
            logit_scale, smooth_eps,
            curr_class_idx, nproc_per_node,
            is_ddp, n_cols, n_rows,
        )

        # Adjust loss for ignore_index
        if ignore_index >= 0:
            loss = loss * (labels != ignore_index).to(loss.dtype)

        # Store loss and lse for backward pass
        ctx.save_for_backward(logits, labels, n_ignore_classes, lse)
        ctx.logit_scale = logit_scale
        ctx.smooth_eps = smooth_eps
        ctx.curr_class_idx = curr_class_idx
        ctx.nproc_per_node = nproc_per_node
        ctx.is_ddp = is_ddp
        ctx.ignore_index = ignore_index
        ctx.is_train = is_train

        return loss

    @staticmethod
    def backward(ctx, dloss):
        logits, labels, n_ignore_classes, lse = ctx.saved_tensors
        n_cols, n_rows = logits.shape

        # Compute the backward pass
        cross_entropy_bwd_wrapper(
            logits, labels, lse,
            dloss,
            ctx.logit_scale, ctx.smooth_eps,
            ctx.curr_class_idx, ctx.nproc_per_node,
            ctx.is_ddp, n_cols, n_rows,
            inplace_backward=not ctx.is_train,
        )

        return logits * 0, None, None, None, None, None, None, None, None, None, None, None, None, None

def cross_entropy_loss(
    logits, labels,
    n_ignore_classes=0,
    logit_scale=1.0,
    smooth_eps
