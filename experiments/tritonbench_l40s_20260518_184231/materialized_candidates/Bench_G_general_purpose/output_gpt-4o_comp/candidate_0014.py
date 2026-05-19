import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr, smoothing, lse_square_scale,
    n_classes, n_samples, ignored_index,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load logits and labels
    logits = tl.load(logits_ptr + offsets * n_classes, mask=offsets < n_samples)
    labels = tl.load(labels_ptr + offsets, mask=offsets < n_samples)

    # Compute LSE
    max_logits = tl.max(logits, axis=1)
    exp_logits = tl.exp(logits - max_logits[:, None])
    lse = tl.log(tl.sum(exp_logits, axis=1)) + max_logits

    # Compute smoothed cross-entropy loss
    one_hot_labels = tl.eye(n_classes, dtype=tl.float32)[labels]
    if smoothing > 0.0:
        one_hot_labels = one_hot_labels * (1 - smoothing) + smoothing / n_classes

    # Mask ignored indices
    mask = labels != ignored_index
    masked_one_hot_labels = one_hot_labels * mask[:, None]

    # Compute loss
    loss = lse - tl.sum(logits * masked_one_hot_labels, axis=1)
    loss = loss * mask  # Ignore specified indices

    # Write back to global memory
    tl.store(loss_ptr + offsets, loss, mask=offsets < n_samples)

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, labels_ptr, grad_output_ptr, grad_logits_ptr, smoothing,
    n_classes, n_samples, ignored_index,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load logits and labels
    logits = tl.load(logits_ptr + offsets * n_classes, mask=offsets < n_samples)
    labels = tl.load(labels_ptr + offsets, mask=offsets < n_samples)
    grad_output = tl.load(grad_output_ptr + offsets, mask=offsets < n_samples)

    # Compute softmax probabilities
    max_logits = tl.max(logits, axis=1)
    exp_logits = tl.exp(logits - max_logits[:, None])
    softmax_probs = exp_logits / tl.sum(exp_logits, axis=1, keepdim=True)

    # Compute one-hot labels
    one_hot_labels = tl.eye(n_classes, dtype=tl.float32)[labels]
    if smoothing > 0.0:
        one_hot_labels = one_hot_labels * (1 - smoothing) + smoothing / n_classes

    # Mask ignored indices
    mask = labels != ignored_index
    masked_one_hot_labels = one_hot_labels * mask[:, None]

    # Compute gradient
    grad_logits = (softmax_probs - masked_one_hot_labels) * grad_output[:, None]
    grad_logits = grad_logits * mask[:, None]  # Ignore specified indices

    # Write back to global memory
    tl.store(grad_logits_ptr + offsets * n_classes, grad_logits, mask=offsets < n_samples)

import torch
from torch.autograd import Function

class CrossEntropyLoss(Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, lse_square_scale=1.0, ignored_index=-1):
        n_samples, n_classes = logits.shape
        loss = torch.empty(n_samples, device=logits.device)

        # Launch Triton forward kernel
        grid = lambda meta: (triton.cdiv(n_samples, meta['BLOCK_SIZE']),)
        cross_entropy_fwd_kernel[grid](
            logits, labels, loss, smoothing, lse_square_scale,
            n_classes, n_samples, ignored_index,
            BLOCK_SIZE=128
        )

        ctx.save_for_backward(logits, labels, loss)
        ctx.smoothing = smoothing
        ctx.ignored_index = ignored_index
        return loss.sum()

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, loss = ctx.saved_tensors
        smoothing = ctx.smoothing
        ignored_index = ctx.ignored_index

        n_samples, n_classes = logits.shape
        grad_logits = torch.empty_like(logits)

        # Launch Triton backward kernel
        grid = lambda meta: (triton.cdiv(n_samples, meta['BLOCK_SIZE']),)
        cross_entropy_bwd_kernel[grid](
            logits, labels, grad_output, grad_logits, smoothing,
            n_classes, n_samples, ignored_index,
            BLOCK_SIZE=128
        )

        return grad_logits, None, None, None, None

# User-friendly wrapper function
def cross_entropy_loss(logits, labels, smoothing=0.0, lse_square_scale=1.0, ignored_index=-1):
    return CrossEntropyLoss.apply(logits, labels, smoothing, lse_square_scale, ignored_index)
