import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_log_softmax(input, target, dim: tl.constexpr, weight, ignore_index: tl.constexpr, reduction: tl.constexpr, label_smoothing: tl.constexpr):
    """
    Computes the cross entropy loss with log softmax applied to the input logits.

    Args:
        input: Input tensor of logits, where softmax will be computed along `dim`.
            The input must be of shape [N, C].
        target: Ground truth class indices or probabilities.
            The target must be of shape [N].
        dim: Dimension along which to compute log softmax.
        weight: Manual rescaling weight for each class.
            Must be of shape [C].
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        label_smoothing: Specifies the amount of smoothing to be applied.

    Returns:
        Loss.
    """
    # Convert input to float32 for numerical stability
    input = input.to(tl.float32)
    target = target.to(tl.int32)

    # Apply log softmax
    max_input = tl.max(input, axis=dim, keepdims=True)
    log_softmax = input - max_input - tl.log(tl.sum(tl.exp(input - max_input), axis=dim, keepdims=True))

    # Handle label smoothing
    if label_smoothing > 0.0:
        num_classes = log_softmax.shape[dim]
        smoothed_target = (1.0 - label_smoothing) * target + (label_smoothing / num_classes)
    else:
        smoothed_target = target

    # Compute cross entropy loss
    loss = -tl.sum(smoothed_target * log_softmax, axis=dim)

    # Apply weight if provided
    if weight is not None:
        loss *= weight[target]

    # Handle reduction
    if reduction == 'mean':
        loss = tl.sum(loss) / tl.sum(target != ignore_index)
    elif reduction == 'sum':
        loss = tl.sum(loss)

    return loss
