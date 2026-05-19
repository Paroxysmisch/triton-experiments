import triton
import triton.language as tl
import torch

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr,            # *F32
    labels_ptr,            # *I32
    loss_ptr,              # *F32
    lse_ptr,               # *F32
    z_loss_ptr,            # *F32
    logit_scale,           # F32
    smoothing,             # F32
    lse_square_scale,      # F32
    ignored_index,         # I32
    total_classes,         # I32
    class_start_idx,       # I32
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr
):
    """
    cross_entropy_fwd_kernel computes the loss, lse and z_loss (optional) for 
    cross-entropy with optional label smoothing and ignoring certain indices.
    Each program processes one row (tl.program_id(0)) and one block of columns 
    (tl.program_id(1)).
    """
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    col_start = block_id * BLOCK_SIZE + class_start_idx

    # Pointers offset:
    offset_row = row_id * total_classes
    logits_ptr += offset_row
    labels_ptr += row_id
    loss_ptr += row_id
    lse_ptr += row_id
    z_loss_ptr += row_id

    # Create column offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    col_ids = col_start + offsets
    mask = col_ids < (class_start_idx + total_classes)

    # Load label
    label_val = tl.load(labels_ptr)
    is_ignored = (label_val == ignored_index)
    # Prepare logits
    logits = tl.load(logits_ptr + col_ids, mask=mask, other=-float('inf'))
    # Optionally apply logit scaling
    logits = logits * logit_scale
    logits = logits.to(tl.float32)

    # Compute max for numerical stability
    max_logits = tl.max(logits, 0)
    # Compute log-sum-exp
    exp_logits = tl.exp(logits - max_logits)
    lse_local = max_logits + tl.log(tl.sum(exp_logits, 0))

    # Write partial log-sum-exp if SPLIT is used, else store final
    if SPLIT:
        # If splitting, each block calculates partial lse. We'll do an atomic add
        # (for demonstration, but real usage might gather tries or separate accumulation).
        # For simplicity, we just store partial sum in lse_ptr (overwriting it).
        tl.atomic_add(lse_ptr, lse_local)
    else:
        # Non-split scenario: each program block processes all columns or partial,
        # but let's assume each row is processed in full by this block_id=0 scenario
        # or we gather partial sums outside. 
        # For demonstration, we store the final LSE only if block_id == 0.
        if block_id == 0:
            tl.store(lse_ptr, lse_local)

    # We only compute the partial loss or final loss if block_id == 0
    if block_id == 0:
        # Retrieve logit for the correct class
        if (label_val >= class_start_idx) & (label_val < (class_start_idx + total_classes)):
            true_class_logit = tl.load(logits_ptr + label_val)
            true_class_logit *= logit_scale
        else:
            true_class_logit = 0.0

        # If the label is ignored, set loss to 0
        loss_val = 0.0
        if not is_ignored:
            if HAS_SMOOTHING:
                # label smoothing: cross-entropy = lse - logit_for_label, adjusted by smoothing
                smooth_term = smoothing / float(total_classes)
                loss_val = (lse_local - true_class_logit) * (1.0 - smoothing) + lse_local * smoothing
                # Another variant is: loss_val = lse_local - (1 - smoothing)*true_class_logit - smooth_term * sum(logits)
            else:
                loss_val = lse_local - true_class_logit

        # Optionally compute z_loss
        z_loss_val = 0.0
        if not is_ignored:
            z_loss_val = (lse_local * lse_local) * lse_square_scale

        tl.store(loss_ptr, loss_val)
        tl.store(z_loss_ptr, z_loss_val)


@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr,       # *F32
    labels_ptr,       # *I32
    dlogits_ptr,      # *F32  (gradient wrt logits)
    loss_ptr,         # *F32  (forward pass loss, used if needed)
    lse_ptr,          # *F32
    logit_scale,      # F32
    smoothing,        # F32
    ignored_index,    # I32
    total_classes,    # I32
    class_start_idx,  # I32
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr
):
    """
    cross_entropy_bwd_kernel computes dlogits for cross-entropy with optional 
    label smoothing, ignoring indices if needed.
    Each program processes one row (tl.program_id(0)) and one block of columns 
    (tl.program_id(1)).
    """
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    col_start = block_id * BLOCK_SIZE + class_start_idx

    # offset pointers
    offset_row = row_id * total_classes
    logits_ptr += offset_row
    dlogits_ptr += offset_row
    labels_ptr += row_id
    lse_ptr += row_id
    loss_ptr += row_id

    offsets = tl.arange(0, BLOCK_SIZE)
    col_ids = col_start + offsets
    mask = col_ids < (class_start_idx + total_classes)

    label_val = tl.load(labels_ptr)
    is_ignored = (label_val == ignored_index)

    # Load log-sum-exp from forward
    lse_val = tl.load(lse_ptr)
    # Load original logits
    logits = tl.load(logits_ptr + col_ids, mask=mask, other=-float('inf'))
    logits = logits * logit_scale
    # Probability
    exp_logits = tl.exp(logits - lse_val)
    # dlogits = p (or p - 1 if it's the true label)
    true_mask = col_ids == label_val

    # If smoothing is used, we reduce the difference for the true class
    # and add a small constant to each class
    grad = exp_logits
    if not is_ignored:
        if HAS_SMOOTHING:
            # Example approach: for smoothing fraction "smoothing" 
            # we add -(1-smoothing) to the true label, +smoothing/total_classes to others
            grad = grad - tl.where(true_mask, 1.0 - smoothing, 0.0) - (smoothing / float(total_classes))
        else:
            grad = grad - tl.where(true_mask, 1.0, 0.0)
    # If label is ignored, gradient is 0
    grad = tl.where(is_ignored, 0.0, grad)

    # scale by logit_scale if used
    grad *= logit_scale

    # store into dlogits
    tl.store(dlogits_ptr + col_ids, grad, mask=mask)


def cross_entropy_fwd(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    ignored_index: int = -100,
    BLOCK_SIZE: int = 128,
    HAS_SMOOTHING: bool = False,
    SPLIT: bool = False,
    class_start_idx: int = 0
):
    """
    Python wrapper for the forward pass of cross entropy. 
    Returns (loss, lse, z_loss).
    Prints intermediate debugging info.
    """
    assert logits.is_cuda, "logits must be a
