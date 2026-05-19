import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_softmax_layernorm_kernel(
    logits_ptr, targets_ptr, out_ptr, loss_ptr,
    logits_stride0, logits_stride1, targets_stride0, targets_stride1,
    n, c, ignore_index, label_smoothing, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_start = pid * BLOCK_SIZE
    batch_end = min(batch_start + BLOCK_SIZE, n)

    # Load logits and targets
    logits = tl.zeros((BLOCK_SIZE, c), dtype=tl.float32)
    targets = tl.zeros((BLOCK_SIZE, c), dtype=tl.float32)
    for i in range(batch_start, batch_end):
        for j in range(c):
            logits[i - batch_start, j] = tl.load(logits_ptr + i * logits_stride0 + j * logits_stride1)
            if targets_ptr is not None:
                targets[i - batch_start, j] = tl.load(targets_ptr + i * targets_stride0 + j * targets_stride1)

    # Apply label smoothing
    if label_smoothing > 0.0:
        targets = (1.0 - label_smoothing) * targets + label_smoothing / c

    # Compute softmax
    logits_max = tl.max(logits, axis=1)
    logits = logits - logits_max[:, None]
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=1)
    softmax_probs = exp_logits / sum_exp_logits[:, None]

    # Compute cross-entropy loss
    loss = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(batch_start, batch_end):
        if targets_ptr is not None:
            loss[i - batch_start] = -tl.sum(targets[i - batch_start] * tl.log(softmax_probs[i - batch_start]))
        else:
            target_idx = tl.load(targets_ptr + i * targets_stride0)
            if target_idx != ignore_index:
                loss[i - batch_start] = -tl.log(softmax_probs[i - batch_start, target_idx])

    # Layer normalization
    mean = tl.sum(softmax_probs, axis=1) / c
    var = tl.sum((softmax_probs - mean[:, None]) ** 2, axis=1) / c
    inv_std = 1.0 / tl.sqrt(var + eps)
    normalized_probs = (softmax_probs - mean[:, None]) * inv_std[:, None]

    # Store results
    for i in range(batch_start, batch_end):
        for j in range(c):
            tl.store(out_ptr + i * logits_stride0 + j * logits_stride1, normalized_probs[i - batch_start, j])
        tl.store(loss_ptr + i, loss[i - batch_start])

import torch
import triton
import triton.language as tl

def fused_cross_entropy_softmax_layernorm(
    logits, targets, normalized_shape, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0, eps=1e-5, out=None
):
    # Validate inputs
    if logits.dim() < 2:
        raise ValueError("Logits must have at least 2 dimensions")
    if targets.dim() < 1:
        raise ValueError("Targets must have at least 1 dimension")
    if logits.size(0) != targets.size(0):
        raise ValueError("Logits and targets must have the same batch size")
    if targets.dim() == 1 and logits.size(1) != targets.size(1):
        raise ValueError("For class indices, targets must have the same number of classes as logits")
    if targets.dim() > 1 and logits.shape != targets.shape:
        raise ValueError("For class probabilities, targets must have the same shape as logits")

    # Prepare output tensors
    if out is None:
        out = torch.empty_like(logits)
    loss = torch.empty(logits.size(0), device=logits.device, dtype=logits.dtype)

    # Launch Triton kernel
    grid = (logits.size(0),)
    block = (128,)
    fused_cross_entropy_softmax_layernorm_kernel[grid, block](
        logits, targets, out, loss,
        logits.stride(0), logits.stride(1), targets.stride(0), targets.stride(1),
        logits.size(0), logits.size(1), ignore_index, label_smoothing, eps,
        BLOCK_SIZE=128
    )

    # Apply reduction
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return out, loss
