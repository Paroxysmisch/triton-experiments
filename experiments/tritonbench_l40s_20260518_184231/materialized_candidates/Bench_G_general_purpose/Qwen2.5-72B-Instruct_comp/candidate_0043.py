import triton
import triton.language as tl

# Define constants
BLOCK_SIZE = 1024
NUM_WARPS = 4

# Forward Kernel
@triton.jit
def _kldiv_kernel_forward(
    y_pred_ptr, y_true_ptr, loss_ptr, 
    BT, V, log_target, reduction, eps, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BT * V

    y_pred = tl.load(y_pred_ptr + offsets, mask=mask)
    y_true = tl.load(y_true_ptr + offsets, mask=mask)

    if log_target:
        loss = tl.exp(y_true) * (y_true - y_pred)
    else:
        y_true_exp = tl.exp(y_true)
        loss = y_true_exp * (y_true - y_pred + tl.log(y_true_exp + eps) - tl.log(tl.exp(y_pred) + eps))

    if reduction == 0:  # "none"
        tl.store(loss_ptr + offsets, loss, mask=mask)
    elif reduction == 1:  # "sum"
        loss_sum = tl.sum(loss, axis=0)
        if pid == 0:
            tl.store(loss_ptr, loss_sum)
    elif reduction == 2:  # "mean"
        loss_sum = tl.sum(loss, axis=0)
        if pid == 0:
            tl.store(loss_ptr, loss_sum / (BT * V))
    elif reduction == 3:  # "batchmean"
        loss_sum = tl.sum(loss, axis=0)
        if pid == 0:
            tl.store(loss_ptr, loss_sum / BT)

# Forward Wrapper
def kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps):
    BT, V = y_pred.shape
    loss = triton.empty((BT * V,) if reduction == "none" else (1,), dtype=y_pred.dtype, device=y_pred.device)
    grid = (triton.cdiv(BT * V, BLOCK_SIZE),)
    _kldiv_kernel_forward[grid](
        y_pred, y_true, loss, BT, V, log_target, 
        0 if reduction == "none" else 1 if reduction == "sum" else 2 if reduction == "mean" else 3, eps, BLOCK_SIZE
    )
    return loss

# Backward Kernel
@triton.jit
def _kldiv_kernel_backward(
    target_ptr, grad_output_ptr, new_grads_ptr, 
    BT, V, log_target, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BT * V

    target = tl.load(target_ptr + offsets, mask=mask)
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)

    if log_target:
        new_grads = tl.exp(target) * (1.0 - tl.exp(grad_output))
    else:
        new_grads = -tl.exp(target) * tl.exp(grad_output)

    tl.store(new_grads_ptr + offsets, new_grads, mask=mask)

# Backward Wrapper
def kldiv_backward_triton(target, grad_output, new_grads, log_target):
    BT, V = target.shape
    grid = (triton.cdiv(BT * V, BLOCK_SIZE),)
    _kldiv_kernel_backward[grid](
        target, grad_output, new_grads, BT, V, log_target, BLOCK_SIZE
    )
    return new_grads

import torch

# Example inputs
BT, V = 1024, 100
y_pred = torch.randn(BT, V, device='cuda')
y_true = torch.randn(BT, V, device='cuda')
log_target = False
reduction = "mean"
eps = 1e-8

# Forward pass
loss = kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps)
print(loss)

# Backward pass
grad_output = torch.ones_like(loss, device='cuda')
new_grads = torch.empty_like(y_pred, device='cuda')
kldiv_backward_triton(y_true, grad_output, new_grads, log_target)
print(new_grads)
