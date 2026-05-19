import torch
import triton
import triton.language as tl

@triton.jit
def cosine_similarity(x1, x2, eps):
    # Compute the dot product between x1 and x2
    dot = tl.sum(x1 * x2, axis=1)
    # Compute the norm of x1 and x2
    norm = tl.sqrt(tl.sum(x1 * x1, axis=1) * tl.sum(x2 * x2, axis=1))
    # Compute cosine similarity
    cos = dot / norm
    cos = cos[:, None]
    # Add eps to avoid division by zero
    cos = cos + eps
    return cos

@triton.jit
def fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size, stride, padding, eps):
    # Compute cosine similarity along dim=1
    cos = cosine_similarity(x1, x2, eps)
    # Add a singleton dimension
    cos = cos.unsqueeze(2)
    # Apply 2D average pooling
    return tl.avg_pool2d(cos, kernel_size, stride, padding)

def wrapper_fused_avg_pool2d_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, eps: float = 1e-8) -> torch.Tensor:
    # Ensure x1 and x2 are 2D tensors
    if x1.dim() != 2 or x2.dim() != 2:
        raise ValueError("Both input tensors must be 2D")
    # Set stride to kernel_size if not provided
    stride = stride if stride is not None else kernel_size
    # Call the Triton kernel
    return fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size, stride, padding, eps)
