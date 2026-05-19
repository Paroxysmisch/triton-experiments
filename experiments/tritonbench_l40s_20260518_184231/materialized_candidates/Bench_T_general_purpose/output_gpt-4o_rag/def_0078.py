import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_log_softmax(input, target, weight, ignore_index, dim, reduction, label_smoothing, output):
    """
    Computes the cross-entropy loss with log-softmax applied to input logits.

    Args:
        input: Input tensor of shape [N, C] containing the logits.
        target: Target tensor of shape [N] containing class indices or probabilities.
        weight: Optional tensor of shape [C] with manual rescaling weights for each class.
        ignore_index: Specifies a target value to be ignored during loss computation.
        dim: Dimension along which to apply log-softmax.
        reduction: Specifies the reduction to apply to the output ('none', 'mean', or 'sum').
        label_smoothing: Amount of label smoothing to apply to the target.
        output: Tensor to store the result of the computed loss.

    Returns:
        The computed loss, possibly reduced, in the `output` tensor.
    """
    
    # Shape of the input
    n, c = input.shape

    # LogSoftmax computation along specified dimension
    input = input.to(tl.float32)
    input_max = tl.max(input, axis=dim, keepdim=True)
    input -= input_max

    exp_input = tl.exp(input)
    logsumexp = tl.log(tl.sum(exp_input, axis=dim, keepdim=True))
    log_softmax = input - logsumexp
    
    # If label smoothing is applied, we adjust the target distribution
    if label_smoothing > 0.0:
        target = target.to(tl.float32)
        target = (1.0 - label_smoothing) * target + label_smoothing / c

    # Cross-entropy loss
    loss = -tl.sum(target * log_softmax, axis=dim)

    # Weighting the loss if provided
    if weight is not None:
        loss = loss * weight[target]

    # Handle the ignore_index (masking out those elements in the target)
    mask = (target != ignore_index).to(tl.float32)
    loss = loss * mask

    # Reduction options: 'none', 'mean', or 'sum'
    if reduction == 'mean':
        loss = tl.sum(loss) / n
    elif reduction == 'sum':
        loss = tl.sum(loss)

    # Write the computed loss to the output
    output[0] = loss

import torch
import triton
import triton.language as tl

def fused_cross_entropy_log_softmax(input: torch.Tensor, target: torch.Tensor, dim: int = 1, weight: torch.Tensor = None, ignore_index: int = -100, reduction: str = 'mean', label_smoothing: float = 0.0) -> torch.Tensor:
    """
    Computes cross-entropy loss with log-softmax applied to the input logits.

    Args:
        input (Tensor): Input tensor of logits of shape (N, C).
        target (Tensor): Ground truth class indices or probabilities of shape (N,).
        dim (int, optional): Dimension along which to compute log-softmax. Default is 1.
        weight (Tensor, optional): Manual rescaling weight for each class.
        ignore_index (int, optional): Specifies a target value to be ignored. Default: -100.
        reduction (str, optional): Specifies the reduction to apply ('none', 'mean', or 'sum'). Default: 'mean'.
        label_smoothing (float, optional): Amount of smoothing applied to target. Default: 0.0.

    Returns:
        Tensor: Computed cross-entropy loss.
    """
    
    # Prepare tensors
    input = input.contiguous()
    target = target.contiguous()
    
    # Prepare an output tensor
    output = torch.empty(1, device=input.device, dtype=torch.float32)

    # Call the Triton kernel
    grid = (input.shape[0],)
    fused_cross_entropy_log_softmax[grid](input, target, weight, ignore_index, dim, reduction, label_smoothing, output)
    
    return output[0]

# Example usage in a script
logits = torch.randn(10, 5, device='cuda')  # 10 samples, 5 classes
targets = torch.randint(0, 5, (10,), device='cuda')  # Ground truth class indices
loss = fused_cross_entropy_log_softmax(logits, targets, dim=1, reduction='mean', label_smoothing=0.1)
print(loss)
