import triton
import triton.language as tl

# Constants for reduction modes
_REDUCTION_MODE_NONE = 0
_REDUCTION_MODE_SUM = 1
_REDUCTION_MODE_MEAN = 2
_REDUCTION_MODE_BATCHMEAN = 3

# Helper function to determine the number of warps
def get_num_warps(block_size):
    if block_size <= 128:
        return 2
    elif block_size <= 256:
        return 4
    else:
        return 8

@triton.jit
def _kldiv_kernel_forward(
    y_pred_ptr, y_true_ptr, output_ptr, n_elements, log_target, reduction, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    y_pred = tl.load(y_pred_ptr + offsets, mask=mask)
    y_true = tl.load(y_true_ptr + offsets, mask=mask)

    if log_target:
        kl_div = y_true * (tl.exp(y_true) - y_pred)
    else:
        kl_div = y_true * (tl.log(y_true) - y_pred)

    if reduction == _REDUCTION_MODE_NONE:
        tl.store(output_ptr + offsets, kl_div, mask=mask)
    elif reduction == _REDUCTION_MODE_SUM:
        output = tl.sum(kl_div, axis=0)
        if pid == 0:
            tl.store(output_ptr, output)
    elif reduction == _REDUCTION_MODE_MEAN:
        output = tl.sum(kl_div, axis=0) / n_elements
        if pid == 0:
            tl.store(output_ptr, output)
    elif reduction == _REDUCTION_MODE_BATCHMEAN:
        output = tl.sum(kl_div, axis=0) / y_true.shape[0]
        if pid == 0:
            tl.store(output_ptr, output)

def kldiv_forward_triton(y_pred, y_true, log_target, reduction):
    n_elements = y_pred.numel()
    output = triton.empty((1,) if reduction != _REDUCTION_MODE_NONE else y_pred.shape, device=y_pred.device, dtype=y_pred.dtype)

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    num_warps = get_num_warps(BLOCK_SIZE)

    _kldiv_kernel_forward[grid](
        y_pred, y_true, output, n_elements, log_target, reduction, BLOCK_SIZE, num_warps=num_warps
    )

    return output

@triton.jit
def _kldiv_kernel_backward(
    input_ptr, target_ptr, grad_output_ptr, grad_input_ptr, n_elements, log_target, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    target = tl.load(target_ptr + offsets, mask=mask)
    grad_output = tl.load(grad_output_ptr, mask=mask)

    if log_target:
        grad_input = -target * tl.exp(target) * grad_output
    else:
        grad_input = -target / input * grad_output

    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)

def kldiv_backward_triton(input, target, grad_output, log_target):
    n_elements = input.numel()
    grad_input = triton.empty_like(input)

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    num_warps = get_num_warps(BLOCK_SIZE)

    _kldiv_kernel_backward[grid](
        input, target, grad_output, grad_input, n_elements, log_target, BLOCK_SIZE, num_warps=num_warps
    )

    return grad_input

import torch

# Example tensors
y_pred = torch.randn(100, 100, device='cuda')
y_true = torch.randn(100, 100, device='cuda').exp()  # Ensure y_true is positive
log_target = False
reduction = _REDUCTION_MODE_MEAN

# Forward pass
output = kldiv_forward_triton(y_pred, y_true, log_target, reduction)
print("KL Divergence:", output)

# Backward pass
grad_output = torch.ones_like(output)  # Example gradient
grad_input = kldiv_backward_triton(y_pred, y_true, grad_output, log_target)
print("Gradient of KL Divergence:", grad_input)
