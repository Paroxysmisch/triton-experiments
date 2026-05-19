import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

@triton.jit
def fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0, eps=1e37, out=None):
    # Compute cross-entropy loss
    loss = cross_entropy_loss(logits, targets, weight, ignore_index, reduction, label_smoothing)
    
    # Apply softmax activation
    probabilities = softmax(logits, eps)
    
    # Apply layer normalization
    normalized_probs = layer_norm(probabilities, normalized_shape, eps)
    
    return loss, normalized_probs

def fused_cross_entropy_softmax_layernorm_wrapper(logits, targets, normalized_shape, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0, eps=1e-5, out=None):
    logits = torch.tensor(logits)
    targets = torch.tensor(targets)
    loss, probabilities = fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, weight, ignore_index, reduction, label_smoothing, eps, out)
    return loss.item(), probabilities

# Example usage
logits = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
targets = [0, 1]
normalized_shape = [1, 2, 3]
logits_tensor = torch.tensor(logits)
targets_tensor = torch.tensor(targets)
loss, probabilities = fused_cross_entropy_softmax_layernorm_wrapper(logits_tensor, targets_tensor, normalized_shape)
print("Loss: ", loss)
print("Probabilities: ", probabilities)
