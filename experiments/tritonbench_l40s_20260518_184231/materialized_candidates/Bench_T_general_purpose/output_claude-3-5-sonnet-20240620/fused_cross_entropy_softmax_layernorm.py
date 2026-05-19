import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_softmax_layernorm_kernel(logits, targets, weight, out, N, C, normalized_shape, ignore_index, label_smoothing, eps):
    # Compute softmax
    max_logits = tl.max(logits, axis=1, keepdims=True)
    exp_logits = tl.exp(logits - max_logits)
    softmax_probs = exp_logits / tl.sum(exp_logits, axis=1, keepdims=True)

    # Compute cross-entropy loss
    if tl.is_tensor(targets, dtype=tl.int32):
        # Class indices
        loss = -tl.sum(tl.log(softmax_probs[tl.arange(0, N), targets]), axis=0)
    else:
        # Class probabilities
        loss = -tl.sum(targets * tl.log(softmax_probs), axis=1)

    # Apply layer normalization
    mean = tl.mean(softmax_probs, axis=1, keepdims=True)
    variance = tl.var(softmax_probs, axis=1, keepdims=True)
    layer_norm_out = (softmax_probs - mean) / tl.sqrt(variance + eps)

    # Apply weight if provided
    if weight is not None:
        layer_norm_out *= weight

    # Store the results
    out.copy_from(layer_norm_out)
    return loss

import torch
from typing import Tuple

def fused_cross_entropy_softmax_layernorm(logits: torch.Tensor, targets: torch.Tensor, normalized_shape: int, 
                                           weight: torch.Tensor = None, ignore_index: int = -100, 
                                           reduction: str = 'mean', label_smoothing: float = 0.0, 
                                           eps: float = 1e-5, *, out: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    N, C = logits.shape
    out = out if out is not None else torch.empty_like(logits)

    # Call the Triton kernel
    loss = fused_cross_entropy_softmax_layernorm_kernel(logits, targets, weight, out, N, C, normalized_shape, 
                                                         ignore_index, label_smoothing, eps)

    # Apply reduction if necessary
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss, out
