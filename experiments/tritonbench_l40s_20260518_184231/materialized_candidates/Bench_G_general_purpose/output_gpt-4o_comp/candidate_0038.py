import triton
import triton.language as tl
import torch

# Forward Kernel: Computes cross-entropy loss, lse, and z_loss
@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr, lse_ptr, z_loss_ptr,
    smoothing, logit_scale, lse_square_scale,
    ignored_index, total_classes, class_start_idx,
    BLOCK_SIZE: tl.constexpr, HAS_SMOOTHING: tl.constexpr, SPLIT: tl.constexpr
):
    row_idx = tl.program_id(0)  # row index
    block_start = tl.program_id(1) * BLOCK_SIZE  # column block start

    # Load logits for this block
    logits = tl.load(logits_ptr + row_idx * total_classes + block_start + tl.arange(0, BLOCK_SIZE), mask=block_start + tl.arange(0, BLOCK_SIZE) < total_classes, other=-float('inf'))
    
    # Scale logits if logit_scale is provided
    if logit_scale != 1.0:
        logits *= logit_scale

    # Compute log-sum-exp (lse)
    max_logits = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logits)
    lse = max_logits + tl.log(tl.sum(exp_logits, axis=0))
    tl.store(lse_ptr + row_idx, lse)

    # Load label and handle ignored index
    label = tl.load(labels_ptr + row_idx)
    if label == ignored_index:
        tl.store(loss_ptr + row_idx, 0.0)
        if SPLIT:
            tl.store(z_loss_ptr + row_idx, 0.0)
        return

    # Compute loss
    true_logit = tl.load(logits_ptr + row_idx * total_classes + label)
    loss = lse - true_logit
    if HAS_SMOOTHING:
        smooth_loss = tl.sum(logits) / total_classes
        loss = (1 - smoothing) * loss + smoothing * smooth_loss
    tl.store(loss_ptr + row_idx, loss)

    # Compute z_loss if SPLIT is enabled
    if SPLIT:
        z_loss = lse_square_scale * (lse ** 2)
        tl.store(z_loss_ptr + row_idx, z_loss)


# Backward Kernel: Computes gradient of logits (dlogits)
@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, labels_ptr, dlogits_ptr, lse_ptr,
    smoothing, logit_scale,
    ignored_index, total_classes, class_start_idx,
    BLOCK_SIZE: tl.constexpr, HAS_SMOOTHING: tl.constexpr
):
    row_idx = tl.program_id(0)  # row index
    block_start = tl.program_id(1) * BLOCK_SIZE  # column block start

    # Load logits and lse
    logits = tl.load(logits_ptr + row_idx * total_classes + block_start + tl.arange(0, BLOCK_SIZE), mask=block_start + tl.arange(0, BLOCK_SIZE) < total_classes, other=0.0)
    lse = tl.load(lse_ptr + row_idx)

    # Compute probabilities
    exp_logits = tl.exp(logits - lse)
    probs = exp_logits / tl.sum(exp_logits, axis=0)

    # Load label and handle ignored index
    label = tl.load(labels_ptr + row_idx)
    if label == ignored_index:
        return

    # Compute dlogits
    one_hot = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    one_hot = tl.where(tl.arange(0, BLOCK_SIZE) == label, 1.0, 0.0)
    dlogits = probs - one_hot
    if HAS_SMOOTHING:
        dlogits = (1 - smoothing) * dlogits + smoothing / total_classes
    if logit_scale != 1.0:
        dlogits *= logit_scale
    tl.store(dlogits_ptr + row_idx * total_classes + block_start + tl.arange(0, BLOCK_SIZE), dlogits)


# Wrapper for Forward Pass
def cross_entropy_fwd(logits, labels, smoothing=0.0, logit_scale=1.0, lse_square_scale=1.0, ignored_index=-1, BLOCK_SIZE=128, HAS_SMOOTHING=False, SPLIT=False):
    batch_size, total_classes = logits.shape
    loss = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)
    lse = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)
    z_loss = torch.empty(batch_size, device=logits.device, dtype=logits.dtype) if SPLIT else None

    grid = (batch_size, (total_classes + BLOCK_SIZE - 1) // BLOCK_SIZE)
    cross_entropy_fwd_kernel[grid](
        logits, labels, loss, lse, z_loss,
        smoothing, logit_scale, lse_square_scale,
        ignored_index, total_classes, 0,
        BLOCK_SIZE=BLOCK_SIZE, HAS_SMOOTHING=HAS_SMOOTHING, SPLIT=SPLIT
    )
    return loss, lse, z_loss


# Wrapper for Backward Pass
def cross_entropy_bwd(logits, labels, lse, smoothing=0.0, logit_scale=1.0, ignored_index=-1, BLOCK_SIZE=128, HAS_SMOOTHING=False):
    batch_size, total_classes = logits.shape
    dlogits = torch.empty_like(logits)

    grid = (batch_size, (total_classes + BLOCK_SIZE - 1) // BLOCK_SIZE)
    cross_entropy_bwd_kernel[grid](
        logits, labels, dlogits, lse,
        smoothing, logit_scale,
        ignored_index, total_classes, 0,
        BLOCK_SIZE=BLOCK_SIZE, HAS_SMOOTHING=HAS_SMOOTHING
    )
    return dlogits
