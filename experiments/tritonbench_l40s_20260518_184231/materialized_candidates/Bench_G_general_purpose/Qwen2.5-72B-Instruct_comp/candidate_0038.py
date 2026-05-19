import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr, lse_ptr, z_loss_ptr,
    total_classes, class_start_idx, ignored_index,
    smoothing, logit_scale, lse_square_scale,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1) * BLOCK_SIZE

    # Load logits for the current row and block
    logits = tl.load(logits_ptr + row_idx * total_classes + col_idx, mask=col_idx + tl.arange(0, BLOCK_SIZE) < total_classes, other=-float('inf'))
    
    # Apply logit scaling
    logits = logits * logit_scale

    # Compute log-sum-exp (lse)
    lse = tl.max(logits, axis=0)
    logits = logits - lse
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    lse = lse + tl.log(sum_exp_logits)

    # Compute loss
    label = tl.load(labels_ptr + row_idx)
    if label == ignored_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx = label - class_start_idx
        if label_idx < 0 or label_idx >= total_classes:
            loss = 0.0
            z_loss = 0.0
        else:
            loss = -logits[label_idx]
            if HAS_SMOOTHING:
                loss = loss * (1 - smoothing) + lse * smoothing
            if SPLIT:
                loss = loss / total_classes
            if lse_square_scale > 0.0:
                z_loss = lse * lse * lse_square_scale
            else:
                z_loss = 0.0

    # Store results
    tl.store(loss_ptr + row_idx, loss)
    tl.store(lse_ptr + row_idx, lse)
    tl.store(z_loss_ptr + row_idx, z_loss)

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, labels_ptr, dlogits_ptr, lse_ptr,
    total_classes, class_start_idx, ignored_index,
    smoothing, logit_scale,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1) * BLOCK_SIZE

    # Load logits for the current row and block
    logits = tl.load(logits_ptr + row_idx * total_classes + col_idx, mask=col_idx + tl.arange(0, BLOCK_SIZE) < total_classes, other=-float('inf'))
    
    # Compute probabilities
    lse = tl.load(lse_ptr + row_idx)
    logits = logits - lse
    exp_logits = tl.exp(logits)
    probs = exp_logits / tl.sum(exp_logits, axis=0)

    # Compute gradients
    label = tl.load(labels_ptr + row_idx)
    if label == ignored_index:
        dlogits = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    else:
        label_idx = label - class_start_idx
        if label_idx < 0 or label_idx >= total_classes:
            dlogits = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        else:
            dlogits = probs
            if HAS_SMOOTHING:
                dlogits = dlogits * (1 - smoothing)
                dlogits[label_idx] -= (1 - smoothing)
            else:
                dlogits[label_idx] -= 1.0
            dlogits = dlogits * logit_scale

    # Store results
    tl.store(dlogits_ptr + row_idx * total_classes + col_idx, dlogits, mask=col_idx + tl.arange(0, BLOCK_SIZE) < total_classes)

import torch

def cross_entropy_fwd(
    logits, labels, smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0,
    ignored_index=-100, total_classes=1000, class_start_idx=0, BLOCK_SIZE=128,
    HAS_SMOOTHING=False, SPLIT=False
):
    # Prepare output tensors
    loss = torch.empty(logits.shape[0], device=logits.device, dtype=logits.dtype)
    lse = torch.empty(logits.shape[0], device=logits.device, dtype=logits.dtype)
    z_loss = torch.empty(logits.shape[0], device=logits.device, dtype=logits.dtype)

    # Dispatch kernel
    grid = (logits.shape[0], (logits.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE)
    cross_entropy_fwd_kernel[grid](
        logits, labels, loss, lse, z_loss,
        total_classes, class_start_idx, ignored_index,
        smoothing, logit_scale, lse_square_scale,
        BLOCK_SIZE, HAS_SMOOTHING, SPLIT
    )

    # Print intermediate results for debugging
    print("Loss:", loss)
    print("LSE:", lse)
    print("Z_loss:", z_loss)

    return loss, lse, z_loss

def cross_entropy_bwd(
    logits, labels, lse, dlogits, logit_scale=1.0,
    ignored_index=-100, total_classes=1000, class_start_idx=0, BLOCK_SIZE=128,
    HAS_SMOOTHING=False
):
    # Dispatch kernel
    grid = (logits.shape[0], (logits.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE)
    cross_entropy_bwd_kernel[grid](
        logits, labels, dlogits, lse,
        total_classes, class_start_idx, ignored_index,
        smoothing, logit_scale,
        BLOCK_SIZE, HAS_SMOOTHING
    )

    # Print intermediate results for debugging
    print("dLogits:", dlogits)

    return dlogits

# Example usage
logits = torch.randn(32, 1000, device='cuda')
labels = torch.randint(0, 1000, (32,), device='cuda')

# Forward pass
loss, lse, z_loss = cross_entropy_fwd(logits, labels, smoothing=0.1, logit_scale=1.0, lse_square_scale=0.01, ignored_index=-100, total_classes=1000, class_start_idx=0, BLOCK_SIZE=128, HAS_SMOOTHING=True, SPLIT=False)

# Backward pass
dlogits = torch.zeros_like(logits)
dlogits = cross_entropy_bwd(logits, labels, lse, dlogits, logit_scale=1.0, ignored_index=-100, total_classes=1000, class_start_idx=0, BLOCK_SIZE=128, HAS_SMOOTHING=True)
