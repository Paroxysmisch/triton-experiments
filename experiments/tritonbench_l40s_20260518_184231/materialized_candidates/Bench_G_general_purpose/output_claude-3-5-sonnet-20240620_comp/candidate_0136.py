import triton
import triton.language as tl
import torch

@triton.jit
def cross_entropy_fwd_kernel(
    # Pointers to matrices
    logits_ptr, labels_ptr, loss_ptr, lse_ptr,
    # Matrix dimensions
    batch_size, num_classes,
    # Optional parameters
    smoothing: tl.float32,
    logits_scale: tl.float32,
    block_id: tl.int32,
    block_size: tl.int32,
    # Strides for the pointers
    stride_logits_batch, stride_logits_class,
    stride_labels,
    BLOCK_SIZE: tl.constexpr
):
    # Position of elements processed by this program
    pid = tl.program_id(0)
    
    # Batch index
    batch_idx = pid
    
    # Bounds checking
    if batch_idx >= batch_size:
        return
        
    # Compute memory offsets for this batch
    logits_offset = batch_idx * stride_logits_batch
    label_offset = batch_idx * stride_labels
    
    # Load label
    label = tl.load(labels_ptr + label_offset)
    
    # Initialize accumulator for log-sum-exp
    max_val = -float('inf')
    class_range = tl.arange(0, BLOCK_SIZE)
    
    # Find maximum for numerical stability
    for class_start in range(0, num_classes, BLOCK_SIZE):
        class_end = min(class_start + BLOCK_SIZE, num_classes)
        mask = class_range < (class_end - class_start)
        logits = tl.load(logits_ptr + logits_offset + (class_start + class_range) * stride_logits_class, mask=mask)
        max_val = tl.maximum(max_val, tl.max(logits, axis=0))
    
    # Compute log-sum-exp
    acc = 0.0
    for class_start in range(0, num_classes, BLOCK_SIZE):
        class_end = min(class_start + BLOCK_SIZE, num_classes)
        mask = class_range < (class_end - class_start)
        logits = tl.load(logits_ptr + logits_offset + (class_start + class_range) * stride_logits_class, mask=mask)
        acc += tl.sum(tl.exp(logits - max_val) * mask)
    
    lse = tl.log(acc) + max_val
    
    # Store log-sum-exp
    tl.store(lse_ptr + batch_idx, lse)
    
    # Compute loss
    target_logit = tl.load(logits_ptr + logits_offset + label * stride_logits_class)
    loss = -target_logit * (1.0 - smoothing) + lse
    
    if smoothing > 0.0:
        loss = loss + smoothing * lse
        
    # Scale loss if needed
    if logits_scale != 1.0:
        loss = loss * logits_scale
        
    # Store loss
    tl.store(loss_ptr + batch_idx, loss)

@triton.jit
def cross_entropy_bwd_kernel(
    # Pointers to matrices
    grad_output_ptr, logits_ptr, labels_ptr, lse_ptr, grad_logits_ptr,
    # Matrix dimensions
    batch_size, num_classes,
    # Optional parameters
    smoothing: tl.float32,
    logits_scale: tl.float32,
    block_id: tl.int32,
    block_size: tl.int32,
    # Strides for the pointers
    stride_logits_batch, stride_logits_class,
    stride_labels,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid
    
    if batch_idx >= batch_size:
        return
        
    # Load gradient and label
    grad = tl.load(grad_output_ptr + batch_idx)
    label = tl.load(labels_ptr + batch_idx * stride_labels)
    lse = tl.load(lse_ptr + batch_idx)
    
    # Compute base gradient scale
    grad_scale = grad * logits_scale
    
    # Process classes in blocks
    class_range = tl.arange(0, BLOCK_SIZE)
    for class_start in range(0, num_classes, BLOCK_SIZE):
        class_end = min(class_start + BLOCK_SIZE, num_classes)
        mask = class_range < (class_end - class_start)
        
        if not tl.sum(mask):
            break
            
        # Load logits for this block
        logits = tl.load(
            logits_ptr + batch_idx * stride_logits_batch + 
            (class_start + class_range) * stride_logits_class,
            mask=mask
        )
        
        # Compute probabilities
        probs = tl.exp(logits - lse)
        
        # Compute gradients
        grad_logits = grad_scale * probs
        
        # Handle label smoothing
        if smoothing > 0.0:
            grad_logits = grad_logits * (1.0 - smoothing) + grad_scale * smoothing * probs
            
        # Adjust gradient for true label
        is_label = (class_start + class_range) == label
        grad_logits = tl.where(
            is_label,
            grad_logits - grad_scale * (1.0 - smoothing),
            grad_logits
        )
        
        # Store gradients
        tl.store(
            grad_logits_ptr + batch_idx * stride_logits_batch +
            (class_start + class_range) * stride_logits_class,
            grad_logits,
            mask=mask
        )

class CrossEntropyLoss:
    def __init__(self, smoothing=0.0, logits_scale=1.0):
        self.smoothing = smoothing
        self.logits_scale = logits_scale
        
    def forward(self, logits, labels):
        batch_size, num_classes = logits.shape
        
        # Allocate output tensors
        loss = torch.empty(batch_size, device=logits.device, dtype=torch.float32)
        lse = torch.empty(batch_size, device=logits.device, dtype=torch.float32)
        
        # Launch kernel
        grid = (triton.cdiv(batch_size, 1),)
        cross_entropy_fwd_kernel[grid](
            logits, labels, loss, lse,
            batch_size, num_classes,
            self.smoothing, self.logits_scale,
            0, num_classes,  # block_id and block_size
            logits.stride(0), logits.stride(1),
            labels.stride(0),
            BLOCK_SIZE=min(128, num_classes)
        )
        
        return loss, lse
        
    def backward(self, grad_output, logits, labels, lse):
        batch_size, num_classes = logits.shape
        
        # Allocate gradient tensor
        grad_logits = torch.empty_like(logits)
        
        # Launch kernel
        grid = (triton.cdiv(batch_size, 1),)
        cross_entropy_bwd_kernel[grid](
            grad_output, logits, labels, lse, grad_logits,
            batch_size, num_classes,
            self.smoothing, self.logits_scale,
            0, num_classes,  # block_id and block_size
            logits.stride(0), logits.stride(1),
            labels.stride(0),
            BLOCK_SIZE=min(128, num_classes)
        )
        
        return grad_logits

def cross_entropy_loss(logits, labels, smoothing=0.0, logits_scale=1.0):
    """
    Compute cross entropy loss with optional label smoothing and logit scaling.
    
    Args:
        logits: Input logits tensor of shape (batch_size, num_classes)
        labels: Ground truth labels tensor of shape (batch_size,)
        smoothing: Label smoothing factor (default: 0.0)
        logits_scale: Scale factor for logits (default: 1.0)
        
    Returns:
        tuple: (loss, lse) tensors of shape (batch_size,)
    """
    loss_fn = CrossEntropyLoss(smoothing=smoothing, logits_scale=logits_scale)
    return loss_fn.forward(logits, labels)
