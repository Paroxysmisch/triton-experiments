import triton
import triton.language as tl
import torch
from typing import Tuple

@triton.jit
def fused_cross_entropy_softmax_layernorm(
    logits, targets, normalized_shape, weight=None, ignore_index=-100, 
    reduction='mean', label_smoothing=0.0, eps=1e-5, out=None
):
    """
    Performs a fused operation combining cross-entropy loss computation,
    softmax activation, and layer normalization.

    Args:
        logits: Input logits of shape (N, C) or (N, C, *).
        targets: Ground truth class indices or class probabilities.
        normalized_shape: Input shape over which layer normalization is applied.
        weight: A manual rescaling weight given to each class (optional).
        ignore_index: Specifies a target value that is ignored (default: -100).
        reduction: Specifies the reduction to apply to the output (default: 'mean').
        label_smoothing: Amount of smoothing when computing the loss (default: 0.0).
        eps: Value added for numerical stability in layer normalization (default: 1e-5).
        out: Output tensor for the normalized probabilities (optional).

    Returns:
        Tuple containing the loss and normalized probabilities.
    """
    # Compute softmax
    logits = logits.to(tl.float32)
    max_logits = tl.max(logits, axis=1, keepdims=True)
    exp_logits = tl.exp(logits - max_logits)
    softmax_probs = exp_logits / tl.sum(exp_logits, axis=1, keepdims=True)

    # Compute cross-entropy loss
    if tl.sum(targets == ignore_index) > 0:
        targets = tl.where(targets == ignore_index, -1, targets)

    if label_smoothing > 0.0:
        num_classes = logits.shape[1]
        targets = (1 - label_smoothing) * targets + label_smoothing / num_classes

    loss = -tl.sum(targets * tl.log(softmax_probs + eps), axis=1)

    if reduction == 'mean':
        loss = tl.mean(loss)
    elif reduction == 'sum':
        loss = tl.sum(loss)

    # Layer normalization
    mean = tl.mean(softmax_probs, axis=1, keepdims=True)
    variance = tl.var(softmax_probs, axis=1, keepdims=True)
    normalized_probs = (softmax_probs - mean) / tl.sqrt(variance + eps)

    if weight is not None:
        normalized_probs = normalized_probs * weight

    if out is not None:
        out.copy_(normalized_probs)

    return loss, normalized_probs
