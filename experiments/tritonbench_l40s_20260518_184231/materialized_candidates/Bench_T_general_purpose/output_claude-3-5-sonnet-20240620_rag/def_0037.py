import triton
import triton.language as tl
import torch

@triton.jit
def fused_cosine_embedding_loss_with_normalization(input1, input2, target, margin: tl.constexpr, reduction: tl.constexpr):
    """
    Computes cosine embedding loss between two normalized tensors.

    Args:
        input1: First input tensor to be normalized and compared.
        input2: Second input tensor to be normalized and compared.
        target: Tensor label with values 1 or -1.
        margin: Margin for dissimilarity. Default: 0.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.

    Returns:
        Computed cosine embedding loss.
    """
    # L2 normalization
    norm1 = input1 / tl.sqrt(tl.sum(input1 * input1, axis=1, keepdims=True) + 1e-8)
    norm2 = input2 / tl.sqrt(tl.sum(input2 * input2, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity
    cosine_similarity = tl.sum(norm1 * norm2, axis=1)

    # Compute loss
    loss = (1 - cosine_similarity) * (target + margin) / 2

    # Apply reduction
    if reduction == 'mean':
        return tl.sum(loss) / loss.shape[0]
    elif reduction == 'sum':
        return tl.sum(loss)
    else:  # 'none'
        return loss

def compute_cosine_embedding_loss(input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor, margin: float = 0, reduction: str = 'mean') -> torch.Tensor:
    """
    Wrapper function for the Triton kernel.

    Args:
        input1 (Tensor): First input tensor to be normalized and compared.
        input2 (Tensor): Second input tensor to be normalized and compared.
        target (Tensor): Tensor label with values 1 or -1.
        margin (float, optional): Margin for dissimilarity. Default: 0.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.

    Returns:
        Tensor: Computed cosine embedding loss.
    """
    # Call the Triton kernel
    return fused_cosine_embedding_loss_with_normalization(input1, input2, target, margin, reduction)
