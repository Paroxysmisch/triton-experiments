import triton
import triton.language as tl

@triton.jit
def fused_kernel(
    logits_ptr, targets_ptr, output_ptr, norm_ptr,
    N, C, normalized_shape, eps, reduction, weight_ptr, ignore_index, label_smoothing,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate row and column indices
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)

    # Load logits and targets
    logits = tl.load(logits_ptr + row_idx * C + col_idx, mask=col_idx < C, other=-float('inf'))
    target = tl.load(targets_ptr + row_idx)

    # Softmax computation
    max_logits = tl.max(logits, axis=0)
    logits = logits - max_logits
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp_logits

    # Cross-entropy loss computation
    if label_smoothing > 0.0:
        nll_loss = -tl.log(probs[target]) * (1.0 - label_smoothing) - label_smoothing / C
    else:
        nll_loss = -tl.log(probs[target])

    # Handle ignore_index
    if target == ignore_index:
        nll_loss = 0.0

    # Reduction
    if reduction == 'mean':
        nll_loss = nll_loss / N
    elif reduction == 'sum':
        nll_loss = tl.sum(nll_loss)

    # Store loss
    tl.store(output_ptr + row_idx, nll_loss)

    # Layer normalization
    mean = tl.sum(probs, axis=0) / normalized_shape
    var = tl.sum((probs - mean) ** 2, axis=0) / normalized_shape
    inv_std = tl.rsqrt(var + eps)
    norm_probs = (probs - mean) * inv_std

    # Store normalized probabilities
    tl.store(norm_ptr + row_idx * C + col_idx, norm_probs, mask=col_idx < C)

import torch

def fused_cross_entropy_softmax_layernorm(
    logits, targets, normalized_shape, weight=None, ignore_index=-100,
    reduction='mean', label_smoothing=0.0, eps=1e-5, *, out=None
):
    N, C = logits.shape[:2]
    logits = logits.contiguous()
    targets = targets.contiguous()

    # Prepare output tensors
    if out is None:
        out = torch.empty_like(logits)

    # Launch Triton kernel
    grid = (N,)
    fused_kernel[grid](
        logits, targets, out, out,
        N, C, normalized_shape, eps, reduction, weight, ignore_index, label_smoothing,
        BLOCK_SIZE=triton.next_power_of_2(C)
    )

    return out
