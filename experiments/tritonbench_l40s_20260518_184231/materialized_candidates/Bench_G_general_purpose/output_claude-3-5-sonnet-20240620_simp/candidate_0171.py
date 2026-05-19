import torch
import triton
import triton.language as tl

@triton.jit
def _cross_entropy_forward_kernel(
    # Pointers to matrices
    logits_ptr, labels_ptr, output_ptr,
    # Matrix dimensions
    batch_size, num_classes,
    # Optional parameters
    softcap_min: tl.float32, softcap_max: tl.float32,
    logit_scale: tl.float32,
    # Strides for memory access
    stride_batch, stride_class,
    # Block size
    BLOCK_SIZE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Batch element this program is responsible for
    batch_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = batch_idx < batch_size
    
    # Initialize max value for stability
    max_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float('inf')
    
    # Load logits and find max for stability
    for c in range(0, num_classes):
        idx = batch_idx * stride_batch + c * stride_class
        logit = tl.load(logits_ptr + idx, mask=mask, other=-float('inf'))
        if logit_scale != 1.0:
            logit = logit * logit_scale
        if softcap_min is not None:
            logit = tl.maximum(logit, softcap_min)
        if softcap_max is not None:
            logit = tl.minimum(logit, softcap_max)
        max_val = tl.maximum(max_val, logit)
    
    # Compute sum of exp(logits - max_val)
    sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for c in range(0, num_classes):
        idx = batch_idx * stride_batch + c * stride_class
        logit = tl.load(logits_ptr + idx, mask=mask, other=-float('inf'))
        if logit_scale != 1.0:
            logit = logit * logit_scale
        if softcap_min is not None:
            logit = tl.maximum(logit, softcap_min)
        if softcap_max is not None:
            logit = tl.minimum(logit, softcap_max)
        sum_exp += tl.exp(logit - max_val)
    
    # Load labels
    labels = tl.load(labels_ptr + batch_idx, mask=mask)
    
    # Compute cross entropy loss
    idx = batch_idx * stride_batch + labels * stride_class
    logit = tl.load(logits_ptr + idx, mask=mask, other=0.0)
    if logit_scale != 1.0:
        logit = logit * logit_scale
    if softcap_min is not None:
        logit = tl.maximum(logit, softcap_min)
    if softcap_max is not None:
        logit = tl.minimum(logit, softcap_max)
    
    loss = -logit + max_val + tl.log(sum_exp)
    
    # Store result
    tl.store(output_ptr + batch_idx, loss, mask=mask)

@triton.jit
def _cross_entropy_backward_kernel(
    # Pointers to matrices
    grad_output_ptr, logits_ptr, labels_ptr, grad_logits_ptr,
    # Matrix dimensions
    batch_size, num_classes,
    # Optional parameters
    softcap_min: tl.float32, softcap_max: tl.float32,
    logit_scale: tl.float32,
    # Strides for memory access
    stride_batch, stride_class,
    # Block size
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = batch_idx < batch_size
    
    # Load gradient from output
    grad_output = tl.load(grad_output_ptr + batch_idx, mask=mask)
    
    # Compute softmax for gradient
    max_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float('inf')
    for c in range(0, num_classes):
        idx = batch_idx * stride_batch + c * stride_class
        logit = tl.load(logits_ptr + idx, mask=mask, other=-float('inf'))
        if logit_scale != 1.0:
            logit = logit * logit_scale
        if softcap_min is not None:
            logit = tl.maximum(logit, softcap_min)
        if softcap_max is not None:
            logit = tl.minimum(logit, softcap_max)
        max_val = tl.maximum(max_val, logit)
    
    sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for c in range(0, num_classes):
        idx = batch_idx * stride_batch + c * stride_class
        logit = tl.load(logits_ptr + idx, mask=mask, other=-float('inf'))
        if logit_scale != 1.0:
            logit = logit * logit_scale
        if softcap_min is not None:
            logit = tl.maximum(logit, softcap_min)
        if softcap_max is not None:
            logit = tl.minimum(logit, softcap_max)
        sum_exp += tl.exp(logit - max_val)
    
    # Load labels
    labels = tl.load(labels_ptr + batch_idx, mask=mask)
    
    # Compute and store gradients
    for c in range(0, num_classes):
        idx = batch_idx * stride_batch + c * stride_class
        logit = tl.load(logits_ptr + idx, mask=mask, other=0.0)
        if logit_scale != 1.0:
            logit = logit * logit_scale
        if softcap_min is not None:
            logit = tl.maximum(logit, softcap_min)
        if softcap_max is not None:
            logit = tl.minimum(logit, softcap_max)
        
        softmax = tl.exp(logit - max_val) / sum_exp
        grad = softmax * grad_output
        if c == labels:
            grad = grad - grad_output
        if logit_scale != 1.0:
            grad = grad * logit_scale
            
        tl.store(grad_logits_ptr + idx, grad, mask=mask)

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, softcap_min=None, softcap_max=None, logit_scale=1.0):
        batch_size, num_classes = logits.shape
        device = logits.device
        output = torch.empty(batch_size, device=device, dtype=torch.float32)
        
        grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE']),)
        
        _cross_entropy_forward_kernel[grid](
            logits, labels, output,
            batch_size, num_classes,
            softcap_min, softcap_max, logit_scale,
            logits.stride(0), logits.stride(1),
            BLOCK_SIZE=128
        )
        
        ctx.save_for_backward(logits, labels)
        ctx.softcap_min = softcap_min
        ctx.softcap_max = softcap_max
        ctx.logit_scale = logit_scale
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        logits, labels = ctx.saved_tensors
        batch_size, num_classes = logits.shape
        device = logits.device
        
        grad_logits = torch.empty_like(logits)
        
        grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE']),)
        
        _cross_entropy_backward_kernel[grid](
            grad_output, logits, labels, grad_logits,
            batch_size, num_classes,
            ctx.softcap_min, ctx.softcap_max, ctx.logit_scale,
            logits.stride(0), logits.stride(1),
            BLOCK_SIZE=128
        )
        
        return grad_logits, None, None, None, None

def fast_cross_entropy_loss(logits, labels, softcap_min=None, softcap_max=None, logit_scale=1.0):
    """
    Compute cross entropy loss using efficient Triton kernels.
    
    Args:
        logits: Input logits of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)
        softcap_min: Minimum value for softcapping logits
        softcap_max: Maximum value for softcapping logits
        logit_scale: Scale factor for logits
    
    Returns:
        Average cross entropy loss
    """
    if len(logits.shape) > 2:
        batch_size = logits.shape[0]
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)
    
    loss = Fast_CrossEntropyLoss.apply(logits, labels, softcap_min, softcap_max, logit_scale)
    return loss.mean()
