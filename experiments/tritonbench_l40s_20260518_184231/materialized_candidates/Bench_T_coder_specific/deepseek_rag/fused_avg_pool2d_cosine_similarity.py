import torch
from torch.nn.functional import avg_pool2d
from torch.nn.functional import cosine_similarity

def fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size, stride=None, padding=0, eps=1e-8):
    if stride is None:
        stride = kernel_size
    # Compute cosine similarity
    cosine_sim = cosine_similarity(x1, x2, dim=1)
    # Add a singleton dimension
    cosine_sim = cosine_sim.unsqueeze(1)
    # Apply 2D average pooling
    avg_pool = avg_pool2d(cosine_sim, kernel_size, stride, padding, eps)
    return avg_pool
