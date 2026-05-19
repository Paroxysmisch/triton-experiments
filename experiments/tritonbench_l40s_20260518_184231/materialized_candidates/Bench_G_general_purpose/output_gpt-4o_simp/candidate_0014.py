import torch
import triton
import triton.language as tl

# Define the forward kernel for cross-entropy loss
@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, output_ptr, lse_ptr,
    n_classes, label_smoothing, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    # Offsets for the current block
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load logits and labels
    logits = tl.load(logits_ptr + offs)
    labels = tl.load(labels_ptr + offs, mask=offs < n_classes)
    
    # Compute Log-Sum-Exp (LSE)
    max_logits = tl.max(logits, axis=0)
    logits = logits - max_logits
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    lse = max_logits + tl.log(sum_exp_logits)
    
    # Store LSE for backward pass
    tl.store(lse_ptr + offs, lse)
    
    # Compute cross-entropy loss with label smoothing
    if label_smoothing > 0:
        smooth_factor = label_smoothing / n_classes
        one_hot_labels = (1 - label_smoothing) * tl.eye(n_classes)[labels] + smooth_factor
    else:
        one_hot_labels = tl.eye(n_classes)[labels]
    
    loss = -tl.sum(one_hot_labels * logits, axis=0) + lse
    tl.store(output_ptr + offs, loss)

# Define the backward kernel for cross-entropy loss
@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, labels_ptr, grad_output_ptr, lse_ptr, grad_logits_ptr,
    n_classes, label_smoothing, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    # Offsets for the current block
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load logits, labels, and LSE
    logits = tl.load(logits_ptr + offs)
    labels = tl.load(labels_ptr + offs, mask=offs < n_classes)
    lse = tl.load(lse_ptr + offs)
    
    # Compute softmax probabilities
    exp_logits = tl.exp(logits - lse)
    probs = exp_logits / tl.sum(exp_logits, axis=0)
    
    # Compute gradients
    if label_smoothing > 0:
        smooth_factor = label_smoothing / n_classes
        one_hot_labels = (1 - label_smoothing) * tl.eye(n_classes)[labels] + smooth_factor
    else:
        one_hot_labels = tl.eye(n_classes)[labels]
    
    grad_logits = probs - one_hot_labels
    grad_output = tl.load(grad_output_ptr + offs)
    grad_logits *= grad_output
    
    # Store gradients
    tl.store(grad_logits_ptr + offs, grad_logits)

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, label_smoothing=0.0):
        n_classes = logits.size(-1)
        batch_size = logits.size(0)
        
        # Allocate output tensors
        output = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)
        lse = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)
        
        # Launch forward kernel
        grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE']),)
        cross_entropy_fwd_kernel[grid](
            logits, labels, output, lse,
            n_classes, label_smoothing, BLOCK_SIZE=128
        )
        
        ctx.save_for_backward(logits, labels, lse)
        ctx.label_smoothing = label_smoothing
        return output.sum()
    
    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, lse = ctx.saved_tensors
        n_classes = logits.size(-1)
        batch_size = logits.size(0)
        
        # Allocate gradient tensor
        grad_logits = torch.empty_like(logits)
        
        # Launch backward kernel
        grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE']),)
        cross_entropy_bwd_kernel[grid](
            logits, labels, grad_output, lse, grad_logits,
            n_classes, ctx.label_smoothing, BLOCK_SIZE=128
        )
        
        return grad_logits, None, None

def cross_entropy_loss(logits, labels, label_smoothing=0.0):
    return CrossEntropyLoss.apply(logits, labels, label_smoothing)
