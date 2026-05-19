{{ code }}
def fused_cross_entropy_log_softmax_kernel(logits, targets, weight, dim, ignore_index, reduction, label_smoothing):
    # Implementation of log softmax
    max_logits = triton.max(logits, dim=dim, keepdim=True)
    stable_logits = logits - max_logits
    log_softmax = stable_logits - triton.log(triton.sum(triton.exp(stable_logits), dim=dim, keepdim=True))

    # Apply label smoothing if specified
    if label_smoothing > 0.0:
        num_classes = logits.shape[dim]
        targets = (1.0 - label_smoothing) * targets + (label_smoothing / num_classes)

    # Compute cross entropy loss
    loss = -triton.sum(targets * log_softmax, dim=dim)

    # Apply weight if specified
    if weight is not None:
        loss *= weight

    # Handle ignore_index
    if ignore_index >= 0:
        loss = triton.where(targets == ignore_index, 0.0, loss)

    # Apply reduction method
    if reduction == 'mean':
        return triton.mean(loss)
    elif reduction == 'sum':
        return triton.sum(loss)
    else:
        return loss

{{ code }}
def fused_cross_entropy_log_softmax(input: torch.Tensor, target: torch.Tensor, dim: int = 1, weight: torch.Tensor = None, ignore_index: int = -100, reduction: str = 'mean', label_smoothing: float = 0.0) -> torch.Tensor:
    # Ensure input and target shapes are compatible
    assert input.dim() == 2 and target.dim() == 1, "Input must be 2D and target must be 1D"
    assert input.size(0) == target.size(0), "Input and target must have the same number of samples"

    # Call the Triton kernel
    return fused_cross_entropy_log_softmax_kernel(input, target, weight, dim, ignore_index, reduction, label_smoothing)
