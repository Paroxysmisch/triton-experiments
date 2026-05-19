import triton
import triton.language as tl

@triton.jit
def cosine_similarity_kernel(x1, x2, output, n_elements, eps):
    # Compute cosine similarity
    idx = tl.arange(0, n_elements)
    x1_norm = tl.sqrt(tl.sum(x1[idx] ** 2, axis=1) + eps)
    x2_norm = tl.sqrt(tl.sum(x2[idx] ** 2, axis=1) + eps)
    dot_product = tl.sum(x1[idx] * x2[idx], axis=1)
    output[idx] = dot_product / (x1_norm * x2_norm)

@triton.jit
def avg_pool2d_kernel(input, output, kernel_size, stride, padding, n_elements):
    # Apply 2D average pooling
    for i in range(0, n_elements, stride):
        for j in range(0, n_elements, stride):
            output[i // stride, j // stride] = tl.sum(input[i:i + kernel_size, j:j + kernel_size]) / (kernel_size * kernel_size)

import torch
import torch.nn.functional as F

def fused_avg_pool2d_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, eps: float = 1e-8) -> torch.Tensor:
    if stride is None:
        stride = kernel_size

    # Compute cosine similarity
    n_elements = x1.size(1)  # Assuming x1 and x2 have the same shape
    output_cosine = torch.empty(x1.size(0), n_elements, device=x1.device)
    cosine_similarity_kernel[(1,)](x1, x2, output_cosine, n_elements, eps)

    # Add singleton dimension
    output_cosine = output_cosine.unsqueeze(1)

    # Apply 2D average pooling
    pooled_output = F.avg_pool2d(output_cosine, kernel_size=kernel_size, stride=stride, padding=padding)

    return pooled_output
