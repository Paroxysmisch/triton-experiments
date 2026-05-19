import triton
import triton.language as tl
import torch
import math

@triton.jit
def cross_entropy_fwd_kernel(
    # Pointers to matrices
    logits_ptr, labels_ptr, loss_ptr, lse_ptr,
    # Matrix dimensions
    batch_size, num_classes,
    # Optional parameters
    smoothing: tl.float32,
    scaling: tl.float32,
    ignored_index: tl.int32,
    # Strides for the different dimensions
    stride_batch, stride_class,
    BLOCK_SIZE: tl.constexpr
):
    # Position of elements processed by this program
    pid = tl.program_id(0)
    
    # Batch index
    batch_idx = pid
    
    # Skip if out of bounds
    if batch_idx >= batch_size:
        return
        
    # Compute loss for this batch element
    label = tl.load(labels_ptr + batch_idx)
    
    # Initialize accumulator for logsumexp
    max_val = float("-inf")
    
    # First pass: find max value for numerical stability
    for c in range(0, num_classes, BLOCK_SIZE):
        mask = c + tl.arange(0, BLOCK_SIZE) < num_classes
        logit = tl.load(logits_ptr + batch_idx * stride_batch + c * stride_class, mask=mask, other=float("-inf"))
        max_val = tl.maximum(max_val, tl.max(logit, axis=0))
    
    # Second pass: compute logsumexp
    acc = 0.0
    for c in range(0, num_classes, BLOCK_SIZE):
        mask = c + tl.arange(0, BLOCK_SIZE) < num_classes
        logit = tl.load(logits_ptr + batch_idx * stride_batch + c * stride_class, mask=mask, other=float("-inf"))
        acc += tl.sum(tl.exp(logit - max_val), axis=0)
    
    log_sum_exp = tl.log(acc) + max_val
    
    # Load logit for the true class
    true_class_logit = tl.load(logits_ptr + batch_idx * stride_batch + label * stride_class)
    
    # Compute cross entropy loss with label smoothing
    if label != ignored_index:
        smooth_factor = smoothing / (num_classes - 1)
        hard_target = 1.0 - smoothing
        loss = -hard_target * true_class_logit + log_sum_exp
        
        # Apply scaling if provided
        if scaling != 1.0:
            loss = loss * scaling
    else:
        loss = 0.0
    
    # Store results
    tl.store(loss_ptr + batch_idx, loss)
    tl.store(lse_ptr + batch_idx, log_sum_exp)

@triton.jit
def cross_entropy_bwd_kernel(
    # Pointers to matrices
    grad_output_ptr, logits_ptr, labels_ptr, grad_logits_ptr,
    lse_ptr,
    # Matrix dimensions
    batch_size, num_classes,
    # Optional parameters
    smoothing: tl.float32,
    scaling: tl.float32,
    ignored_index: tl.int32,
    # Strides for the different dimensions
    stride_batch, stride_class,
    BLOCK_SIZE: tl.constexpr
):
    # Position of elements processed by this program
    pid = tl.program_id(0)
    
    # Batch and class indices
    batch_idx = pid // num_classes
    class_idx = pid % num_classes
    
    # Skip if out of bounds
    if batch_idx >= batch_size:
        return
        
    # Load values
    label = tl.load(labels_ptr + batch_idx)
    logit = tl.load(logits_ptr + batch_idx * stride_batch + class_idx * stride_class)
    lse = tl.load(lse_ptr + batch_idx)
    grad_output = tl.load(grad_output_ptr + batch_idx)
    
    # Compute gradient
    if label != ignored_index:
        prob = tl.exp(logit - lse)
        smooth_factor = smoothing / (num_classes - 1)
        
        if class_idx == label:
            grad = (prob - (1.0 - smoothing)) * grad_output
        else:
            grad = (prob - smooth_factor) * grad_output
            
        if scaling != 1.0:
            grad = grad * scaling
    else:
        grad = 0.0
    
    # Store gradient
    tl.store(grad_logits_ptr + batch_idx * stride_batch + class_idx * stride_class, grad)

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, scaling=1.0, ignored_index=-100):
        batch_size, num_classes = logits.shape
        device = logits.device
        
        # Allocate output tensors
        loss = torch.empty(batch_size, device=device, dtype=torch.float32)
        lse = torch.empty(batch_size, device=device, dtype=torch.float32)
        
        # Launch kernel
        grid = (batch_size,)
        cross_entropy_fwd_kernel[grid](
            logits, labels, loss, lse,
            batch_size, num_classes,
            smoothing, scaling, ignored_index,
            logits.stride(0), logits.stride(1),
            BLOCK_SIZE=128,
        )
        
        # Save for backward
        ctx.save_for_backward(logits, labels, lse)
        ctx.smoothing = smoothing
        ctx.scaling = scaling
        ctx.ignored_index = ignored_index
        
        return loss.mean(), lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        logits, labels, lse = ctx.saved_tensors
        batch_size, num_classes = logits.shape
        
        # Allocate output tensor
        grad_logits = torch.empty_like(logits)
        
        # Launch kernel
        grid = (batch_size * num_classes,)
        cross_entropy_bwd_kernel[grid](
            grad_output, logits, labels, grad_logits, lse,
            batch_size, num_classes,
            ctx.smoothing, ctx.scaling, ctx.ignored_index,
            logits.stride(0), logits.stride(1),
            BLOCK_SIZE=128,
        )
        
        return grad_logits, None, None, None, None

def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.0,
    scaling: float = 1.0,
    ignored_index: int = -100
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cross entropy loss with label smoothing and scaling.
    
    Args:
        logits: Input logits of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)
        smoothing: Label smoothing factor (default: 0.0)
        scaling: Loss scaling factor (default: 1.0)
        ignored_index: Index to ignore in the loss computation (default: -100)
    
    Returns:
        tuple: (loss, lse) where loss is the mean cross entropy loss and
              lse is the log-sum-exp values for each batch element
    """
    return CrossEntropyLoss.apply(logits, labels, smoothing, scaling, ignored_index)
