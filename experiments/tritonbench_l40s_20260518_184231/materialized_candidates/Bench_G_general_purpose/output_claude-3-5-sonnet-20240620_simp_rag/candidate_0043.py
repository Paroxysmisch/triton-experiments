import triton
import triton.language as tl
import torch

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr, lse_ptr, z_loss_ptr,
    stride, total_classes, class_start_idx,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute pointers
    logits_offset = pid * stride + class_start_idx
    
    # Load block of logits
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < (total_classes - class_start_idx)
    logits = tl.load(logits_ptr + logits_offset + col_offsets, mask=mask, other=-float('inf'))
    
    # Load label
    label = tl.load(labels_ptr + pid)
    
    # Compute max for numerical stability
    max_logit = tl.max(logits, axis=0)
    
    # Compute log-sum-exp
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    log_sum_exp = tl.log(sum_exp) + max_logit
    
    # Compute loss
    label_logit = tl.load(logits_ptr + logits_offset + label) if label != -100 else 0.0
    loss = log_sum_exp - label_logit
    
    # Optional label smoothing
    if HAS_SMOOTHING:
        smooth_factor = 0.1
        loss = (1.0 - smooth_factor) * loss + smooth_factor * log_sum_exp
    
    # Compute z_loss (stability regularization)
    z_loss = 1e-4 * (log_sum_exp * log_sum_exp)
    
    # Store results
    tl.store(loss_ptr + pid, loss)
    tl.store(lse_ptr + pid, log_sum_exp)
    tl.store(z_loss_ptr + pid, z_loss)

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, dloss_ptr, dlogits_ptr, lse_ptr, labels_ptr,
    stride, total_classes, class_start_idx,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute pointers
    offset = pid * stride + class_start_idx
    
    # Load values
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < (total_classes - class_start_idx)
    logits = tl.load(logits_ptr + offset + col_offsets, mask=mask, other=-float('inf'))
    dloss = tl.load(dloss_ptr + pid)
    lse = tl.load(lse_ptr + pid)
    label = tl.load(labels_ptr + pid)
    
    # Compute softmax gradients
    exp_logits = tl.exp(logits - lse)
    dlogits = exp_logits
    
    # Adjust for true label
    if label != -100:
        label_mask = col_offsets == (label - class_start_idx)
        dlogits = tl.where(label_mask, exp_logits - 1.0, exp_logits)
    
    # Apply label smoothing if enabled
    if HAS_SMOOTHING:
        smooth_factor = 0.1
        dlogits = (1.0 - smooth_factor) * dlogits + smooth_factor * exp_logits
    
    # Scale by loss gradient and store
    dlogits = dlogits * dloss
    tl.store(dlogits_ptr + offset + col_offsets, dlogits, mask=mask)

def cross_entropy_fwd(logits, labels, smoothing=0.0, logit_scale=1.0, lse_square_scale=1e-4):
    # Get shapes
    batch_size = logits.shape[0]
    total_classes = logits.shape[1]
    
    # Allocate output tensors
    loss = torch.empty(batch_size, device=logits.device, dtype=torch.float32)
    lse = torch.empty_like(loss)
    z_loss = torch.empty_like(loss)
    
    # Configure kernel
    BLOCK_SIZE = min(256, total_classes)
    grid = (batch_size,)
    
    # Launch kernel
    cross_entropy_fwd_kernel[grid](
        logits, labels, loss, lse, z_loss,
        logits.stride(0), total_classes, 0,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_SMOOTHING=smoothing > 0,
        SPLIT=(total_classes > BLOCK_SIZE)
    )
    
    return loss, lse, z_loss

def cross_entropy_bwd(dlogits, dloss, logits, lse, labels, smoothing=0.0):
    # Get shapes
    batch_size = logits.shape[0]
    total_classes = logits.shape[1]
    
    # Configure kernel
    BLOCK_SIZE = min(256, total_classes)
    grid = (batch_size,)
    
    # Launch kernel
    cross_entropy_bwd_kernel[grid](
        logits, dloss, dlogits, lse, labels,
        logits.stride(0), total_classes, 0,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_SMOOTHING=smoothing > 0
    )
    
    return dlogits

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, logit_scale=1.0, lse_square_scale=1e-4):
        loss, lse, z_loss = cross_entropy_fwd(
            logits, labels, smoothing, logit_scale, lse_square_scale
        )
        ctx.save_for_backward(logits, labels, lse)
        ctx.smoothing = smoothing
        return loss, lse, z_loss

    @staticmethod
    def backward(ctx, dloss, dlse, dz_loss):
        logits, labels, lse = ctx.saved_tensors
        dlogits = torch.zeros_like(logits)
        dlogits = cross_entropy_bwd(
            dlogits, dloss, logits, lse, labels, ctx.smoothing
        )
        return dlogits, None, None, None, None
