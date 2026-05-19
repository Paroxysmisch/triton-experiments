import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr, lse_ptr,  # pointers to data
    n_classes, smoothing, scale, ignore_index,  # parameters
    BLOCK_SIZE: tl.constexpr
):
    # Define block index
    block_idx = tl.program_id(0)
    
    # Compute the offset for this block
    start_idx = block_idx * BLOCK_SIZE
    
    # Load logits and labels
    logits = tl.load(logits_ptr + start_idx, mask=start_idx < n_classes)
    labels = tl.load(labels_ptr + block_idx, mask=block_idx < n_classes)

    # Compute log-sum-exp for numerical stability
    max_logits = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    lse = max_logits + tl.log(sum_exp_logits)
    
    # Store log-sum-exp
    tl.store(lse_ptr + block_idx, lse, mask=block_idx < n_classes)

    # Compute cross-entropy loss
    correct_logits = logits[labels]
    loss = lse - correct_logits

    # Apply label smoothing if specified
    if smoothing > 0.0:
        loss = (1.0 - smoothing) * loss + smoothing * lse / n_classes

    # Apply scaling if specified
    if scale != 1.0:
        loss *= scale

    # Ignore certain indices
    if labels == ignore_index:
        loss = 0.0

    # Store the loss
    tl.store(loss_ptr + block_idx, loss, mask=block_idx < n_classes)

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, labels_ptr, grads_ptr, d_logits_ptr,  # pointers to data
    n_classes, smoothing, scale, ignore_index,  # parameters
    BLOCK_SIZE: tl.constexpr
):
    # Define block index
    block_idx = tl.program_id(0)
    
    # Compute the offset for this block
    start_idx = block_idx * BLOCK_SIZE
    
    # Load logits and labels
    logits = tl.load(logits_ptr + start_idx, mask=start_idx < n_classes)
    labels = tl.load(labels_ptr + block_idx, mask=block_idx < n_classes)
    
    # Load the gradients
    grads = tl.load(grads_ptr + block_idx, mask=block_idx < n_classes)

    # Compute probabilities
    max_logits = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp_logits

    # Compute gradient with respect to logits
    d_logits = probs
    d_logits[labels] -= 1.0

    # Apply label smoothing if specified
    if smoothing > 0.0:
        d_logits = (1.0 - smoothing) * d_logits + smoothing / n_classes

    # Apply scaling if specified
    if scale != 1.0:
        d_logits *= scale

    # Ignore certain indices
    if labels == ignore_index:
        d_logits = 0.0

    # Store the gradients
    tl.store(d_logits_ptr + start_idx, d_logits * grads, mask=start_idx < n_classes)

import torch

class CrossEntropyLoss:
    def __init__(self, smoothing=0.0, scale=1.0, ignore_index=-1):
        self.smoothing = smoothing
        self.scale = scale
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        # Allocate memory for outputs
        loss = torch.empty_like(labels, dtype=logits.dtype)
        lse = torch.empty_like(labels, dtype=logits.dtype)
        
        # Launch the forward kernel
        grid = (labels.numel(),)
        cross_entropy_fwd_kernel[grid](
            logits, labels, loss, lse,
            logits.shape[-1], self.smoothing, self.scale, self.ignore_index,
            BLOCK_SIZE=128
        )
        
        return loss, lse

    def backward(self, logits, labels, grad_output):
        # Allocate memory for gradients
        d_logits = torch.empty_like(logits)
        
        # Launch the backward kernel
        grid = (labels.numel(),)
        cross_entropy_bwd_kernel[grid](
            logits, labels, grad_output, d_logits,
            logits.shape[-1], self.smoothing, self.scale, self.ignore_index,
            BLOCK_SIZE=128
        )
        
        return d_logits

def cross_entropy_loss(logits, labels, smoothing=0.0, scale=1.0, ignore_index=-1):
    loss_fn = CrossEntropyLoss(smoothing, scale, ignore_index)
    return loss_fn.forward(logits, labels)
