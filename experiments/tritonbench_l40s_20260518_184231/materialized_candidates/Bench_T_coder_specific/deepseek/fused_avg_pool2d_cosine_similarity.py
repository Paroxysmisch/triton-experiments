import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

def fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size, stride=None, padding=0, eps=1e-8):
    if stride is None:
        stride = kernel_size

    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)

    # Compute cosine similarity
    cos_sim = F.cosine_similarity(x1, x2, dim=1, eps=eps).unsqueeze(1)

    # Apply 2D average pooling
    avg_pool = F.avg_pool2d(cos_sim, kernel_size, stride, padding, eps)

    return avg_pool
