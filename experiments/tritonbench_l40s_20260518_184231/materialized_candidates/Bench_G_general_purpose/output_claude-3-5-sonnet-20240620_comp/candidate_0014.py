import torch
import triton
import triton.language as tl
import math

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr, lse_ptr,
    stride_b, stride_c,
    n_classes, batch_size,
    smoothing: tl.float32,
    lse_square_scale: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute batch index and offsets
    batch_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = batch_idx < batch_size
    
    # Initialize accumulators for max and sum
    max_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float('inf')
    lse_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute max for numerical stability
    for c in range(0, n_classes):
        logit_idx = batch_idx[:, None] * stride_b + c * stride_c
        logits = tl.load(logits_ptr + logit_idx, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, logits)
    
    # Compute LSE (Log Sum Exp)
    for c in range(0, n_classes):
        logit_idx = batch_idx[:, None] * stride_b + c * stride_c
        logits = tl.load(logits_ptr + logit_idx, mask=mask, other=-float('inf'))
        lse_sum += tl.exp(logits - max_val)
    
    lse = tl.log(lse_sum) + max_val
    
    # Load true labels
    labels = tl.load(labels_ptr + batch_idx, mask=mask)
    
    # Compute cross entropy with label smoothing
    true_logits = tl.load(
        logits_ptr + batch_idx * stride_b + labels * stride_c,
        mask=mask,
        other=0.0
    )
    
    smooth_factor = smoothing / n_classes
    loss = (1.0 - smoothing) * (-true_logits + lse) + smooth_factor * (n_classes * lse)
    
    # Store results
    tl.store(loss_ptr + batch_idx, loss, mask=mask)
    tl.store(lse_ptr + batch_idx, lse, mask=mask)

@triton.jit
def cross_entropy_bwd_kernel(
    grad_output_ptr, logits_ptr, labels_ptr, lse_ptr, grad_input_ptr,
    stride_b, stride_c,
    n_classes, batch_size,
    smoothing: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = batch_idx < batch_size
    
    # Load LSE and labels
    lse = tl.load(lse_ptr + batch_idx, mask=mask)
    labels = tl.load(labels_ptr + batch_idx, mask=mask)
    grad_output = tl.load(grad_output_ptr + batch_idx, mask=mask)
    
    # Compute gradients for all classes
    smooth_factor = smoothing / n_classes
    for c in range(0, n_classes):
        logit_idx = batch_idx * stride_b + c * stride_c
        logits = tl.load(logits_ptr + logit_idx, mask=mask)
        
        # Compute probability
        prob = tl.exp(logits - lse)
        
        # Compute gradient
        is_true_class = (labels == c)
        grad = (prob - (1.0 - smoothing) * is_true_class - smooth_factor) * grad_output
        
        # Store gradient
        tl.store(grad_input_ptr + logit_idx, grad, mask=mask)

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, lse_square_scale=1.0):
        batch_size, n_classes = logits.shape
        device = logits.device
        
        # Allocate output tensors
        loss = torch.empty(batch_size, device=device, dtype=torch.float32)
        lse = torch.empty(batch_size, device=device, dtype=torch.float32)
        
        # Launch kernel
        grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE']),)
        cross_entropy_fwd_kernel[grid](
            logits, labels, loss, lse,
            logits.stride(0), logits.stride(1),
            n_classes, batch_size,
            smoothing, lse_square_scale,
            BLOCK_SIZE=128
        )
        
        # Save for backward
        ctx.save_for_backward(logits, labels, lse)
        ctx.smoothing = smoothing
        
        return loss.mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, lse = ctx.saved_tensors
        batch_size, n_classes = logits.shape
        device = logits.device
        
        # Allocate gradient tensor
        grad_input = torch.empty_like(logits)
        
        # Launch backward kernel
        grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE']),)
        cross_entropy_bwd_kernel[grid](
            grad_output.expand(batch_size), logits, labels, lse, grad_input,
            logits.stride(0), logits.stride(1),
            n_classes, batch_size,
            ctx.smoothing,
            BLOCK_SIZE=128
        )
        
        return grad_input, None, None, None

def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.0,
    lse_square_scale: float = 1.0
) -> torch.Tensor:
    """
    User-friendly wrapper for cross entropy loss computation using Triton.
    
    Args:
        logits: Input tensor of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)
        smoothing: Label smoothing factor (default: 0.0)
        lse_square_scale: LSE regularization scale (default: 1.0)
    
    Returns:
        Scalar loss value
    """
    return CrossEntropyLoss.apply(logits, labels, smoothing, lse_square_scale)
