import torch
import torch.nn.functional as F

def fused_avg_pool2d_cosine_similarity(
    x1: torch.Tensor,
    x2: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes the cosine similarity between x1 and x2 along dim=1, adds a singleton dimension,
    and applies 2D average pooling.

    Parameters
