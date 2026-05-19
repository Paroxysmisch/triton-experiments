import triton
import triton.language as tl

# Constants
BLOCK_SIZE = 256
NUM_WARPS = 4

# Forward Kernel
@triton.jit
def _kldiv_kernel_forward(
    y_pred_ptr, y_true_ptr, output_ptr, 
    BT, V, log_target, reduction, eps,
    stride_y_pred_BT, stride_y_pred_V, 
    stride_y_true_BT, stride_y_true_V, 
    stride_output_BT, stride_output_V,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BT * V

    y_pred_offsets = tl.zeros((BLOCK_SIZE,), tl.int32) + offsets
    y_true_offsets = tl.zeros((BLOCK_SIZE,), tl.int32) + offsets
    output_offsets = tl.zeros((BLOCK_SIZE,), tl.int32) + offsets

    y_pred = tl.load(y_pred_ptr + y_pred_offsets, mask=mask, other=0.0)
    y_true = tl.load(y_true_ptr + y_true_offsets, mask=mask, other=0.0)

    if log_target:
        y_true = tl.exp(y_true)

    kl_div = y_true * (tl.log(y_true + eps) - y_pred)
    tl.store(output_ptr + output_offsets, kl_div, mask=mask)

    if reduction == "sum":
        block_sum = tl.sum(kl_div, axis=0)
        tl.atomic_add(output_ptr, block_sum)
    elif reduction == "mean":
        block_sum = tl.sum(kl_div, axis=0)
        block_mean = block_sum / (BT * V)
        tl.atomic_add(output_ptr, block_mean)
    elif reduction == "batchmean":
        block_sum = tl.sum(kl_div, axis=0)
        block_mean = block_sum / BT
        tl.atomic_add(output_ptr, block_mean)

# Forward Wrapper
def kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps):
    BT, V = y_pred.shape
    output = tl.zeros((BT, V), dtype=tl.float32)

    grid = (BT * V + BLOCK_SIZE - 1) // BLOCK_SIZE
    _kldiv_kernel_forward[grid](
        y_pred, y_true, output,
        BT, V, log_target, reduction, eps,
        y_pred.stride(0), y_pred.stride(1),
        y_true.stride(0), y_true.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )

    if reduction in ["sum", "mean", "batchmean"]:
        output = output.sum()

    return output

# Backward Kernel
@triton.jit
def _kldiv_kernel_backward(
    target_ptr, grad_output_ptr, new_grads_ptr,
    BT, V, log_target,
    stride_target_BT, stride_target_V,
    stride_grad_output_BT, stride_grad_output_V,
    stride_new_grads_BT, stride_new_grads_V,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BT * V

    target_offsets = tl.zeros((BLOCK_SIZE,), tl.int32) + offsets
    grad_output_offsets = tl.zeros((BLOCK_SIZE,), tl.int32) + offsets
    new_grads_offsets = tl.zeros((BLOCK_SIZE,), tl.int32) + offsets

    target = tl.load(target_ptr + target_offsets, mask=mask, other=0.0)
    grad_output = tl.load(grad_output_ptr + grad_output_offsets, mask=mask, other=0.0)

    if log_target:
        target = tl.exp(target)

    new_grads = -target * grad_output
    tl.store(new_grads_ptr + new_grads_offsets, new_grads, mask=mask)

# Backward Wrapper
def kldiv_backward_triton(target, grad_output, log_target):
    BT, V = target.shape
    new_grads = tl.zeros((BT, V), dtype=tl.float32)

    grid = (BT * V + BLOCK_SIZE - 1) // BLOCK_SIZE
    _kldiv_kernel_backward[grid](
        target, grad_output, new_grads,
        BT, V, log_target,
        target.stride(0), target.stride(1),
        grad_output.stride(0), grad_output.stride(1),
        new_grads.stride(0), new_grads.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )

    return new_grads

import torch

# Example usage
BT, V = 1024, 128
y_pred = torch.randn(BT, V, device='cuda')
y_true = torch.randn(BT, V, device='cuda').softmax(dim=1)  # Ensure y_true is a valid probability distribution
log_target = False
reduction = "mean"
eps = 1e-8

# Forward pass
loss = kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps)
print("Loss:", loss)

# Backward pass
grad_output = torch.ones_like(loss, device='cuda')
grads = kldiv_backward_triton(y_true, grad_output, log_target)
print("Gradients:", grads)
