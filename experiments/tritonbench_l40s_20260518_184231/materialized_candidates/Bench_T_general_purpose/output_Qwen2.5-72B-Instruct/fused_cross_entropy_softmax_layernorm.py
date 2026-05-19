import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_softmax_layernorm_kernel(
    logits_ptr, targets_ptr, output_ptr, loss_ptr,
    N, C, normalized_shape, weight_ptr, ignore_index, reduction, label_smoothing, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * C

    # Load logits and targets
    logits = tl.load(logits_ptr + offsets, mask=mask)
    targets = tl.load(targets_ptr + offsets, mask=mask)

    # Compute softmax
    logits_exp = tl.exp(logits)
    logits_exp_sum = tl.sum(logits_exp, axis=1)
    softmax = logits_exp / logits_exp_sum

    # Compute cross-entropy loss
    if label_smoothing > 0.0:
        smooth_targets = (1 - label_smoothing) * targets + label_smoothing / C
        loss = -tl.sum(smooth_targets * tl.log(softmax), axis=1)
    else:
        loss = -tl.log(softmax[tl.arange(0, N), targets])

    # Apply weight and reduction
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + targets, mask=mask)
        loss = loss * weight

    if reduction == 'mean':
        loss = tl.sum(loss) / N
    elif reduction == 'sum':
        loss = tl.sum(loss)

    # Apply ignore_index
    if ignore_index >= 0:
        loss = tl.where(targets == ignore_index, 0.0, loss)

    # Store loss
    tl.store(loss_ptr + pid, loss, mask=mask)

    # Compute layer normalization
    mean = tl.sum(softmax, axis=1) / normalized_shape
    var = tl.sum((softmax - mean) ** 2, axis=1) / normalized_shape
    normalized = (softmax - mean) / tl.sqrt(var + eps)

    # Store normalized output
    tl.store(output_ptr + offsets, normalized, mask=mask)

import torch
import triton
import triton.language as tl

def fused_cross_entropy_softmax_layernorm(
    logits, targets, normalized_shape, weight=None, ignore_index=-100, reduction='mean',
    label_smoothing=0.0, eps=1e-5, *, out=None
):
    N, C = logits.shape[:2]
    if out is None:
        out = torch.empty_like(logits)

    # Flatten logits and targets for kernel
    logits_flat = logits.view(-1)
    targets_flat = targets.view(-1)

    # Allocate memory for loss
    loss = torch.empty(1, device=logits.device)

    # Define grid and block sizes
    BLOCK_SIZE = 1024
    grid = (N * C + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Define weight tensor if provided
    weight_ptr = weight.data_ptr() if weight is not None else 0

    # Call Triton kernel
    fused_cross_entropy_softmax_layernorm_kernel[
        grid, BLOCK_SIZE
    ](
        logits_flat, targets_flat, out.view(-1), loss,
        N, C, normalized_shape, weight_ptr, ignore_index, reduction, label_smoothing, eps
    )

    return out, loss
