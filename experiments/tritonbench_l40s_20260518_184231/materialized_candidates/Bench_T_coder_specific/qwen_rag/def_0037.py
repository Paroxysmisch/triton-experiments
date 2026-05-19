import triton
import triton.language as tl
import torch

@triton.jit
def l2_normalize(x, y, dim: tl.constexpr):
    """L2 normalizes the input tensor along the specified dimension."""
    x_norm = tl.norm(x, ord=2, dim=dim)
    y_norm = tl.norm(y, ord=2, dim=dim)
    x_normalized = x / x_norm[:, None]
    y_normalized = y / y_norm[:, None]
    return x_normalized, y_normalized

@triton.jit
def cosine_similarity(x, y):
    """Computes the cosine similarity between two tensors."""
    dot_product = tl.dot(x, y.T)
    return dot_product / (tl.norm(x, ord=2, dim=1) * tl.norm(y, ord=2, dim=1))

@triton.jit
def cosine_embedding_loss(cos_sim, target, margin, reduction: tl.constexpr):
    """Computes the cosine embedding loss."""
    loss = tl.maximum(0, target * cos_sim - margin)
    if reduction == 'mean':
        return tl.mean(loss)
    elif reduction == 'sum':
        return tl.sum(loss)
    else:
        return loss

@triton.jit
def fused_cosine_embedding_loss_with_normalization(input1, input2, target, margin: tl.constexpr = 0, reduction: tl.constexpr = 'mean'):
    """Fused cosine embedding loss with normalization."""
    x_normalized, y_normalized = l2_normalize(input1, input2, dim=1)
    cos_sim = cosine_similarity(x_normalized, y_normalized)
    loss = cosine_embedding_loss(cos_sim, target, margin, reduction)
    return loss

# Wrapper function
def fused_cosine_embedding_loss_with_normalization_wrapper(input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor, margin: float = 0, reduction: str = 'mean') -> torch.Tensor:
    # Convert PyTorch tensors to Triton tensors
    x_triton = tl.tensor(input1.numpy(), dtype=tl.float32)
    y_triton = tl.tensor(input2.numpy(), dtype=tl.float32)
    target_triton = tl.tensor(target.numpy(), dtype=tl.float32)

    # Call the Triton kernel
    loss_triton = fused_cosine_embedding_loss_with_normalization(x_triton, y_triton, target_triton, margin, reduction)

    # Convert the Triton tensor back to PyTorch tensor
    loss_pytorch = torch.from_numpy(loss_triton.numpy())

    return loss_pytorch
