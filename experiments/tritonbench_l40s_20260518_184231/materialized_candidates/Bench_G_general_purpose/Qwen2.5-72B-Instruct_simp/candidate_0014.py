import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr, lse_ptr, smoothing, ignored_index, n_classes, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    logits = tl.load(logits_ptr + offsets, mask=mask, other=-float('inf'))
    labels = tl.load(labels_ptr + offsets, mask=mask, other=0)

    # Compute log-sum-exp (LSE)
    max_logits = tl.max(logits, axis=0)
    logits_sub_max = logits - max_logits
    exp_logits = tl.exp(logits_sub_max)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    lse = max_logits + tl.log(sum_exp_logits)

    # Compute cross-entropy loss
    loss = tl.where(labels == ignored_index, 0.0, -logits[labels] + lse)
    if smoothing > 0.0:
        loss += smoothing * (lse - tl.log(n_classes))

    tl.store(loss_ptr + offsets, loss, mask=mask)
    tl.store(lse_ptr + offsets, lse, mask=mask)

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, labels_ptr, lse_ptr, grad_loss_ptr, grad_logits_ptr, smoothing, ignored_index, n_classes, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    logits = tl.load(logits_ptr + offsets, mask=mask, other=-float('inf'))
    labels = tl.load(labels_ptr + offsets, mask=mask, other=0)
    lse = tl.load(lse_ptr + offsets, mask=mask, other=0.0)
    grad_loss = tl.load(grad_loss_ptr + offsets, mask=mask, other=0.0)

    # Compute gradients
    exp_logits = tl.exp(logits - lse)
    grad = exp_logits * grad_loss
    grad = tl.where(labels == ignored_index, 0.0, grad)
    if smoothing > 0.0:
        grad += smoothing * (grad_loss - grad_loss / n_classes)

    tl.store(grad_logits_ptr + offsets, grad, mask=mask)

import torch
import triton
import triton.runtime

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, ignored_index=-100, n_classes=1000, distributed=False):
        logits = logits.contiguous()
        labels = labels.contiguous()
        n_elements = logits.numel() // n_classes

        # Allocate memory for loss and LSE
        loss = torch.empty_like(logits)
        lse = torch.empty_like(logits)

        # Define block size
        BLOCK_SIZE = 1024

        # Launch forward kernel
        grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        cross_entropy_fwd_kernel[grid, BLOCK_SIZE](
            logits, labels, loss, lse, smoothing, ignored_index, n_classes, n_elements
        )

        ctx.save_for_backward(logits, labels, lse)
        ctx.smoothing = smoothing
        ctx.ignored_index = ignored_index
        ctx.n_classes = n_classes
        ctx.distributed = distributed

        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        logits, labels, lse = ctx.saved_tensors
        n_elements = logits.numel() // ctx.n_classes

        # Allocate memory for gradients
        grad_logits = torch.empty_like(logits)

        # Define block size
        BLOCK_SIZE = 1024

        # Launch backward kernel
        grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        cross_entropy_bwd_kernel[grid, BLOCK_SIZE](
            logits, labels, lse, grad_loss, grad_logits, ctx.smoothing, ctx.ignored_index, ctx.n_classes, n_elements
        )

        return grad_logits, None, None, None, None, None

def cross_entropy_loss(logits, labels, smoothing=0.0, ignored_index=-100, n_classes=1000, distributed=False):
    return CrossEntropyLoss.apply(logits, labels, smoothing, ignored_index, n_classes, distributed)

import torch

# Example inputs
logits = torch.randn(128, 1000, device='cuda')
labels = torch.randint(0, 1000, (128,), device='cuda')

# Compute loss
loss = cross_entropy_loss(logits, labels, smoothing=0.1, ignored_index=-100, n_classes=1000)

# Backward pass
loss.mean().backward()
