import triton
import triton.language as tl
import torch

# Constants
BLOCK_SIZE = 128

@triton.jit
def cross_entropy_fwd_kernel(
    # Pointers to tensors
    logits_ptr, labels_ptr, loss_ptr, lse_ptr, z_loss_ptr,
    # Scalar values
    smoothing, logit_scale, lse_square_scale, ignored_index,
    # Shape information
    batch_size, total_classes, class_start_idx,
    # Strides for memory access
    logits_batch_stride, logits_class_stride,
    # Features
    HAS_SMOOTHING: tl.constexpr, SPLIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for batch and class block
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    # Calculate offsets
    class_range = tl.arange(0, BLOCK_SIZE)
    class_ids = class_start_idx + block_idx * BLOCK_SIZE + class_range
    
    # Load logits for current block
    logits_offset = row_idx * logits_batch_stride + class_ids * logits_class_stride
    mask = class_ids < total_classes
    logits = tl.load(logits_ptr + logits_offset, mask=mask, other=-float('inf'))
    
    # Scale logits
    logits = logits * logit_scale
    
    # Load label for current row
    label = tl.load(labels_ptr + row_idx)
    
    # Calculate log-sum-exp
    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    lse = tl.log(sum_exp) + max_logit
    
    # Store LSE
    tl.store(lse_ptr + row_idx, lse)
    
    # Calculate loss
    is_ignored = label == ignored_index
    if HAS_SMOOTHING:
        smooth_factor = smoothing / total_classes
        label_logit = tl.where(class_ids == label, 1.0 - smoothing + smooth_factor, smooth_factor)
    else:
        label_logit = tl.where(class_ids == label, 1.0, 0.0)
    
    loss = -tl.sum(label_logit * logits, axis=0) + lse
    
    # Calculate z_loss if needed
    z_loss = lse * lse * lse_square_scale
    
    # Store results
    if not is_ignored:
        tl.store(loss_ptr + row_idx, loss)
        tl.store(z_loss_ptr + row_idx, z_loss)
    else:
        tl.store(loss_ptr + row_idx, 0.0)
        tl.store(z_loss_ptr + row_idx, 0.0)

@triton.jit
def cross_entropy_bwd_kernel(
    # Pointers to tensors
    logits_ptr, labels_ptr, lse_ptr, dlogits_ptr,
    # Scalar values
    smoothing, logit_scale, ignored_index,
    # Shape information
    batch_size, total_classes, class_start_idx,
    # Strides
    logits_batch_stride, logits_class_stride,
    # Features
    HAS_SMOOTHING: tl.constexpr, SPLIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    # Calculate offsets
    class_range = tl.arange(0, BLOCK_SIZE)
    class_ids = class_start_idx + block_idx * BLOCK_SIZE + class_range
    
    # Load data
    logits_offset = row_idx * logits_batch_stride + class_ids * logits_class_stride
    mask = class_ids < total_classes
    logits = tl.load(logits_ptr + logits_offset, mask=mask, other=-float('inf'))
    label = tl.load(labels_ptr + row_idx)
    lse = tl.load(lse_ptr + row_idx)
    
    # Calculate probabilities
    logits = logits * logit_scale
    probs = tl.exp(logits - lse)
    
    # Calculate gradients
    if HAS_SMOOTHING:
        smooth_factor = smoothing / total_classes
        label_probs = tl.where(class_ids == label, 1.0 - smoothing + smooth_factor, smooth_factor)
        dlogits = (probs - label_probs) * logit_scale
    else:
        dlogits = probs * logit_scale
        dlogits = tl.where(class_ids == label, dlogits - logit_scale, dlogits)
    
    # Handle ignored index
    is_ignored = label == ignored_index
    if is_ignored:
        dlogits = tl.zeros_like(dlogits)
    
    # Store gradients
    tl.store(dlogits_ptr + logits_offset, dlogits, mask=mask)

def cross_entropy_fwd(logits, labels, smoothing=0.0, logit_scale=1.0, 
                     lse_square_scale=0.0, ignored_index=-100):
    batch_size, total_classes = logits.shape
    
    # Allocate output tensors
    loss = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)
    lse = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)
    z_loss = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)
    
    # Configure grid
    num_blocks = triton.cdiv(total_classes, BLOCK_SIZE)
    grid = (batch_size, num_blocks)
    
    # Launch kernel
    cross_entropy_fwd_kernel[grid](
        logits, labels, loss, lse, z_loss,
        smoothing, logit_scale, lse_square_scale, ignored_index,
        batch_size, total_classes, 0,
        logits.stride(0), logits.stride(1),
        smoothing > 0.0, False, BLOCK_SIZE
    )
    
    return loss, lse, z_loss

def cross_entropy_bwd(logits, labels, lse, smoothing=0.0, logit_scale=1.0, 
                     ignored_index=-100):
    batch_size, total_classes = logits.shape
    
    # Allocate gradient tensor
    dlogits = torch.empty_like(logits)
    
    # Configure grid
    num_blocks = triton.cdiv(total_classes, BLOCK_SIZE)
    grid = (batch_size, num_blocks)
    
    # Launch kernel
    cross_entropy_bwd_kernel[grid](
        logits, labels, lse, dlogits,
        smoothing, logit_scale, ignored_index,
        batch_size, total_classes, 0,
        logits.stride(0), logits.stride(1),
        smoothing > 0.0, False, BLOCK_SIZE
    )
    
    return dlogits
