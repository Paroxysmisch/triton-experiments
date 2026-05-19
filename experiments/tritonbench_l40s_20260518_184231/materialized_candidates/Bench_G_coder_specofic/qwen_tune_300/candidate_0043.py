import torch
import triton
import triton.language as tl

@triton.jit
def _kldiv_kernel_forward(
    y_true,
    y_pred,
    loss,
    log_target,
    reduction,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the KL divergence loss
    batch_idx = tl.program_id(0)
    y_true_row = tl.load(y_true + batch_idx)
    y_pred_row = tl.load(y_pred + batch_idx * BLOCK_SIZE)
    end_idx = tl.where(reduction == "none", y_pred_row + BLOCK_SIZE, y_pred_row + 1)
    for idx in range(y_pred_row, end_idx):
        y_pred_val = tl.load(y_pred + idx, mask=idx < end_idx, other=0.0)
        y_true_val = tl.where(log_target, y_true_row, tl.exp(y_true_row))
        if log_target:
            loss_val = y_true_val * (tl.log(y_true_val) - y_pred_val)
        else:
            loss_val = y_true_val * (tl.log(y_true_val) - y_pred_val)
        if reduction == "none":
            tl.store(loss + batch_idx * BLOCK_SIZE + idx, loss_val, mask=idx < end_idx)
        else:
            loss_val = tl.where(idx < end_idx, loss_val, 0.0)
            loss = tl.sum(loss_val)
            if reduction == "batchmean":
                loss = loss / y_pred_row.shape[0]
            elif reduction == "mean":
                loss = loss / (y_pred_row.shape[0] * BLOCK_SIZE)
            elif reduction == "sum":
                pass
            loss = tl.sum(loss)
            tl.store(loss + batch_idx, loss)

def kldiv_forward_triton(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    log_target: bool,
    reduction: str,
    eps: float = 1e-8,
):
    # Wrapper function for forward KL divergence computation
    loss = torch.empty(
        y_true.shape if reduction == "none" else (1 if reduction == "batchmean" else (y_true.shape[0],)),
        device=y_pred.device,
    )
    assert reduction in ["none", "sum", "mean", "batchmean"]
    BLOCK_SIZE = triton.next_power_of_2(y_pred.shape[1])
    num_warps = 4
    _kldiv_kernel_forward[(y_true.shape[0],)](
        y_true,
        y_pred,
        loss,
        log_target,
        reduction,
        eps,
        BLOCK_SIZE,
        num_warps=num_warps,
    )
    return loss

@triton.jit
def _kldiv_kernel_backward(
    grad_output,
    target,
    y_pred,
    new_grads,
    log_target,
    reduction,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute gradients for backward pass
    batch_idx = tl.program_id(0)
    grad_output_val = tl.load(grad_output + batch_idx)
    y_pred_row = tl.load(y_pred + batch_idx * BLOCK_SIZE)
    target_val = tl.load(target + batch_idx)
    end_idx = tl.where(reduction == "none", y_pred_row + BLOCK_SIZE, y_pred_row + 1)
    for idx in range(y_pred_row, end_idx):
        y_pred_val = tl.load(y_pred + idx, mask=idx < end_idx, other=0.0)
        if log_target:
            loss_val = grad_output_val * (tl.exp(target_val) - y_pred_val)
        else:
            loss_val = grad_output_val * (target_val - y_pred_val)
        tl.store(new_grads + idx, loss_val, mask=idx < end_idx)

def kldiv_backward_triton(
    target: torch.Tensor,
    grad_output: torch.Tensor,
    new_grads: torch.Tensor,
    log_target: bool,
    reduction: str,
    eps: float = 1e-8,
):
    # Wrapper function for backward KL divergence computation
    assert reduction in ["none", "sum", "mean", "batchmean"]
    BLOCK_SIZE = triton.next_power_of_2(target.shape[1])
    num_warps = 4
    _kldiv_kernel_backward[(target.shape[0],)](
        grad_output,
        target,
        new_grads,
        log_target,
        reduction,
        eps,
        BLOCK_SIZE,
        num_warps=num_warps,
    )
    return new_grads
