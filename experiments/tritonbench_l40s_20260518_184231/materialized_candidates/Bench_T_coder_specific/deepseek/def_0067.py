import triton
import torch

def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float = 2.0, eps: float = 1e0-6, keepdim: bool = False) -> torch.Tensor:
    # Apply adaptive average pooling to x1 and x2
    x1 = triton.language.functional.adaptive_avg_pool2d(x1, output_size)
    x2 = triton.language.functional.adaptive_avg_pool2d(x2, output_size)

    # Compute pairwise distance between the pooled outputs
    distance = triton.language.functional.pairwise_distance(x1, x2, p=p, eps=eps, keepdim=keepdim)

    return distance
