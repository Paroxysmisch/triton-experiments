import torch
import triton
import triton.language as tl
import math

@triton.jit
def cross_entropy_fwd_kernel(
    # Pointers to matrices
    logits_ptr, labels_ptr, lse_ptr, loss_ptr,
    # Matrix dimensions
    batch_size, num_classes,
    # Parameters
    label_smoothing,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute batch element this program is responsible for
    batch_idx = pid
    
    if batch_idx >= batch_size:
        return
        
    # Compute log-sum-exp (LSE)
    max_val = float('-inf')
    logits_row_ptr = logits_ptr + batch_idx * num_classes
    
    # Find max for numerical stability
    for i in range(0, num_classes, BLOCK_SIZE):
        mask = i + tl.arange(0, BLOCK_SIZE) < num_classes
        logits = tl.load(logits_row_ptr + i, mask=mask, other=float('-inf'))
        max_val = tl.maximum(max_val, tl.max(logits, axis=0))
    
    # Compute exp sum
    exp_sum = 0.0
    for i in range(0, num_classes, BLOCK_SIZE):
        mask = i + tl.arange(0, BLOCK_SIZE) < num_classes
        logits = tl.load(logits_row_ptr + i, mask=mask, other=float('-inf'))
        exp_sum += tl.sum(tl.exp(logits - max_val), axis=0)
    
    lse = tl.log(exp_sum) + max_val
    tl.store(lse_ptr + batch_idx, lse)
    
    # Compute cross entropy loss
    label_idx = tl.load(labels_ptr + batch_idx)
    logit = tl.load(logits_row_ptr + label_idx)
    
    if label_smoothing > 0.0:
        smooth_loss = -lse
        one_hot_loss = -logit + lse
        loss = (1.0 - label_smoothing) * one_hot_loss + label_smoothing * smooth_loss
    else:
        loss = -logit + lse
        
    tl.store(loss_ptr + batch_idx, loss)

@triton.jit
def cross_entropy_bwd_kernel(
    # Pointers to matrices
    grad_output_ptr, logits_ptr, labels_ptr, lse_ptr, grad_logits_ptr,
    # Matrix dimensions
    batch_size, num_classes,
    # Parameters
    label_smoothing,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute indices
    batch_idx = pid
    
    if batch_idx >= batch_size:
        return
        
    # Load values
    grad_output = tl.load(grad_output_ptr + batch_idx)
    lse = tl.load(lse_ptr + batch_idx)
    label_idx = tl.load(labels_ptr + batch_idx)
    
    # Compute gradients for all logits in this row
    logits_row_ptr = logits_ptr + batch_idx * num_classes
    grad_logits_row_ptr = grad_logits_ptr + batch_idx * num_classes
    
    for i in range(0, num_classes, BLOCK_SIZE):
        mask = i + tl.arange(0, BLOCK_SIZE) < num_classes
        logits = tl.load(logits_row_ptr + i, mask=mask, other=0.0)
        
        if label_smoothing > 0.0:
            # Gradient with label smoothing
            probs = tl.exp(logits - lse)
            grad = grad_output * ((1.0 - label_smoothing) * probs)
            
            # Add smoothing gradient
            smooth_grad = -label_smoothing * grad_output / num_classes
            grad = grad + smooth_grad
            
            # Adjust gradient for true class
            true_class_mask = i + tl.arange(0, BLOCK_SIZE) == label_idx
            grad = tl.where(true_class_mask, 
                          grad - grad_output * (1.0 - label_smoothing),
                          grad)
        else:
            # Standard cross entropy gradient
            probs = tl.exp(logits - lse)
            grad = grad_output * probs
            
            # Adjust gradient for true class
            true_class_mask = i + tl.arange(0, BLOCK_SIZE) == label_idx
            grad = tl.where(true_class_mask, grad - grad_output, grad)
            
        tl.store(grad_logits_row_ptr + i, grad, mask=mask)

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, label_smoothing=0.0):
        batch_size, num_classes = logits.shape
        device = logits.device
        
        # Allocate output tensors
        lse = torch.empty(batch_size, device=device, dtype=logits.dtype)
        loss = torch.empty(batch_size, device=device, dtype=logits.dtype)
        
        # Define block size
        BLOCK_SIZE = 32
        
        # Launch kernel
        grid = (batch_size,)
        cross_entropy_fwd_kernel[grid](
            logits, labels, lse, loss,
            batch_size, num_classes,
            label_smoothing,
            BLOCK_SIZE
        )
        
        # Save for backward
        ctx.save_for_backward(logits, labels, lse)
        ctx.label_smoothing = label_smoothing
        ctx.num_classes = num_classes
        
        return loss.mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, lse = ctx.saved_tensors
        batch_size, num_classes = logits.shape
        label_smoothing = ctx.label_smoothing
        
        # Allocate gradient tensors
        grad_logits = torch.empty_like(logits)
        
        # Define block size
        BLOCK_SIZE = 32
        
        # Launch kernel
        grid = (batch_size,)
        cross_entropy_bwd_kernel[grid](
            grad_output.expand(batch_size), logits, labels, lse, grad_logits,
            batch_size, num_classes,
            label_smoothing,
            BLOCK_SIZE
        )
        
        return grad_logits, None, None

def cross_entropy_loss(logits, labels, label_smoothing=0.0):
    """
    Compute cross entropy loss with optional label smoothing.
    
    Args:
        logits: Tensor of shape (batch_size, num_classes) containing the logits
        labels: Tensor of shape (batch_size,) containing the target labels
        label_smoothing: Float in [0, 1] for label smoothing factor
    
    Returns:
        Scalar loss value
    """
    return CrossEntropyLoss.apply(logits, labels, label_smoothing)
