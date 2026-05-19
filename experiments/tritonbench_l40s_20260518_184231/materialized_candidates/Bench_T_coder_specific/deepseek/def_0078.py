import torch
import torch.nn.functional as F

def fused_cross_entropy_log_softmax(input: torch.Tensor, target: torch.Tensor, dim: int = 1, weight: torch.Tensor = None, ignore_index: int = -100, reduction: str = 'mean', label_smoothing: float = 0.0) -> torch.Tensor:
    if label_smoothing > 0.0:
        n_classes = input.size(dim)
        target = target + label_smoothing * torch.randn_like(target) * (1 - 1 / n_classes)
        target = torch.clamp(target, 0, 1)

    log_probs = F.log_softmax(input, dim=dim)
    loss = F.nll_loss(log_probs, target, weight=weight, ignore_index=ignore_index, reduction=reduction)

    return loss
