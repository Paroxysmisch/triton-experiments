import triton
import triton.language as tl

@triton.jit
def fused_cosine_embedding_loss_with_normalization(input1, input2, target, margin: tl.constexpr, reduction: tl.constexpr):
    """
    Computes the cosine embedding loss between two normalized tensors, encouraging similarity when the target is 1 and dissimilarity when the target is -1.
    
    Args:
        input1: First input tensor to be normalized and compared. Shape: [BATCH_SIZE, FEATURE_DIM].
        input2: Second input tensor to be normalized and compared. Shape: [BATCH_SIZE, FEATURE_DIM].
        target: Tensor with labels of 1 or -1, indicating whether to encourage similarity or dissimilarity. Shape: [BATCH_SIZE].
        margin: Optional margin for dissimilarity. Default is 0.
        reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'. Default is 'mean'.

    Returns:
        The cosine embedding loss after reduction.
    """
    input1 = input1.to(tl.float32)
    input2 = input2.to(tl.float32)
    target = target.to(tl.float32)

    # L2 Normalization of input1 and input2 along dimension 1
    input1_norm = input1 / tl.norm(input1, axis=1, keepdim=True)
    input2_norm = input2 / tl.norm(input2, axis=1, keepdim=True)

    # Compute cosine similarity
    cosine_similarity = tl.sum(input1_norm * input2_norm, axis=1)

    # Compute loss based on target values
    loss = (1 - target) * 0.5 * tl.maximum(0., cosine_similarity - margin) + target * 0.5 * tl.maximum(0., -cosine_similarity - margin)

    # Apply reduction method
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return tl.sum(loss) / loss.shape[0]
    elif reduction == 'sum':
        return tl.sum(loss)

    return loss  # Default return (shouldn't hit here)
