import triton
import triton.language as tl
import torch

@triton.jit
def cross_entropy_fwd_kernel(
    # Pointers to matrices
    logits_ptr, labels_ptr, loss_ptr, lse_ptr, z_loss_ptr,
    # Matrix dimensions
    batch_size, total_classes, class_start_idx,
    # Parameters
    smoothing, logit_scale, lse_square_scale, ignored_index,
    # Constants
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute the row index
    row_idx = pid
    
    # Handle out of bounds
    if row_idx >= batch_size:
        return
        
    # Compute memory offsets
    logits_offset = row_idx * total_classes
    
    # Initialize accumulators
    max_val = float("-inf")
    sum_exp = 0.0
    loss_val = 0.0
    
    # First pass: find max for numerical stability
    for col in range(0, total_classes, BLOCK_SIZE):
        col_offset = tl.arange(0, BLOCK_SIZE)
        mask = col_offset < (total_classes - col)
        logits = tl.load(logits_ptr + logits_offset + col + col_offset, mask=mask, other=float("-inf"))
        max_val = tl.maximum(max_val, tl.max(logits, axis=0))
    
    # Second pass: compute softmax and cross entropy
    for col in range(0, total_classes, BLOCK_SIZE):
        col_offset = tl.arange(0, BLOCK_SIZE)
        mask = col_offset < (total_classes - col)
        
        # Load logits and labels
        logits = tl.load(logits_ptr + logits_offset + col + col_offset, mask=mask, other=0.0)
        labels = tl.load(labels_ptr + logits_offset + col + col_offset, mask=mask, other=0.0)
        
        # Apply logit scaling if needed
        if logit_scale != 1.0:
            logits = logits * logit_scale
            
        # Compute exp(logits - max_val) for numerical stability
        exp_val = tl.exp(logits - max_val)
        sum_exp += tl.sum(exp_val, axis=0)
        
        # Compute cross entropy loss
        if HAS_SMOOTHING:
            smooth_target = smoothing / total_classes
            hard_target = 1.0 - smoothing
            target = tl.where(labels > 0, hard_target, smooth_target)
        else:
            target = labels
            
        loss_val -= tl.sum(target * (logits - max_val), axis=0)
    
    # Compute final loss and LSE
    log_sum_exp = tl.log(sum_exp) + max_val
    loss_val += log_sum_exp
    
    # Compute z_loss if needed
    z_loss = lse_square_scale * (log_sum_exp * log_sum_exp)
    
    # Store results
    tl.store(loss_ptr + row_idx, loss_val)
    tl.store(lse_ptr + row_idx, log_sum_exp)
    tl.store(z_loss_ptr + row_idx, z_loss)

@triton.jit
def cross_entropy_bwd_kernel(
    # Pointers to matrices
    logits_ptr, labels_ptr, dloss_ptr, dlogits_ptr,
    # Matrix dimensions
    batch_size, total_classes,
    # Parameters
    smoothing, logit_scale, ignored_index,
    # Constants
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute row index
    row_idx = pid
    
    # Handle out of bounds
    if row_idx >= batch_size:
        return
        
    # Compute memory offsets
    offset = row_idx * total_classes
    
    # Load dloss
    dloss = tl.load(dloss_ptr + row_idx)
    
    # Compute softmax and gradients for the entire row
    max_val = float("-inf")
    
    # Find max for numerical stability
    for col in range(0, total_classes, BLOCK_SIZE):
        col_offset = tl.arange(0, BLOCK_SIZE)
        mask = col_offset < (total_classes - col)
        logits = tl.load(logits_ptr + offset + col + col_offset, mask=mask, other=float("-inf"))
        max_val = tl.maximum(max_val, tl.max(logits, axis=0))
    
    # Compute softmax and gradients
    sum_exp = 0.0
    exp_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # First pass: compute sum of exponentials
    for col in range(0, total_classes, BLOCK_SIZE):
        col_offset = tl.arange(0, BLOCK_SIZE)
        mask = col_offset < (total_classes - col)
        logits = tl.load(logits_ptr + offset + col + col_offset, mask=mask, other=0.0)
        exp_vals = tl.exp(logits - max_val)
        sum_exp += tl.sum(exp_vals, axis=0)
    
    # Second pass: compute gradients
    for col in range(0, total_classes, BLOCK_SIZE):
        col_offset = tl.arange(0, BLOCK_SIZE)
        mask = col_offset < (total_classes - col)
        
        # Load values
        logits = tl.load(logits_ptr + offset + col + col_offset, mask=mask, other=0.0)
        labels = tl.load(labels_ptr + offset + col + col_offset, mask=mask, other=0.0)
        
        # Compute probabilities
        probs = tl.exp(logits - max_val) / sum_exp
        
        # Compute gradients
        if HAS_SMOOTHING:
            smooth_target = smoothing / total_classes
            hard_target = 1.0 - smoothing
            target = tl.where(labels > 0, hard_target, smooth_target)
            dlogits = (probs - target) * dloss
        else:
            dlogits = (probs - labels) * dloss
            
        # Apply logit scaling if needed
        if logit_scale != 1.0:
            dlogits = dlogits * logit_scale
            
        # Store gradients
        tl.store(dlogits_ptr + offset + col + col_offset, dlogits, mask=mask)

def cross_entropy_fwd(logits, labels, smoothing=0.0, logit_scale=1.0, 
                     lse_square_scale=0.0, ignored_index=-100, 
                     BLOCK_SIZE=1024):
    batch_size, total_classes = logits.shape
    device = logits.device
    
    # Allocate output tensors
    loss = torch.empty(batch_size, device=device, dtype=torch.float32)
    lse = torch.empty(batch_size, device=device, dtype=torch.float32)
    z_loss = torch.empty(batch_size, device=device, dtype=torch.float32)
    
    # Configure grid and block sizes
    grid = (batch_size,)
    
    # Launch kernel
    cross_entropy_fwd_kernel[grid](
        logits, labels, loss, lse, z_loss,
        batch_size, total_classes, 0,
        smoothing, logit_scale, lse_square_scale, ignored_index,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_SMOOTHING=(smoothing > 0.0),
        SPLIT=(total_classes > BLOCK_SIZE)
    )
    
    return loss, lse, z_loss

def cross_entropy_bwd(logits, labels, dloss, smoothing=0.0, 
                     logit_scale=1.0, ignored_index=-100, 
                     BLOCK_SIZE=1024):
    batch_size, total_classes = logits.shape
    device = logits.device
    
    # Allocate output tensor
    dlogits = torch.empty_like(logits)
    
    # Configure grid
    grid = (batch_size,)
    
    # Launch kernel
    cross_entropy_bwd_kernel[grid](
        logits, labels, dloss, dlogits,
        batch_size, total_classes,
        smoothing, logit_scale, ignored_index,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_SMOOTHING=(smoothing > 0.0)
    )
    
    return dlogits
