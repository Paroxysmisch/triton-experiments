import triton
import triton.language as tl
import torch

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, 
    labels_ptr, 
    loss_ptr, 
    lse_ptr, 
    z_loss_ptr,
    smoothing, 
    logit_scale, 
    lse_square_scale,
    ignored_index, 
    total_classes,
    class_start_idx,
    stride_row_logits, 
    stride_col_logits,
    stride_labels, 
    stride_row_loss, 
    stride_row_lse, 
    stride_row_zloss,
    BLOCK_SIZE: tl.constexpr, 
    HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr
):
    row_id = tl.program_id(0)
    col_block_id = tl.program_id(1)

    row_offset_logits = row_id * stride_row_logits
    row_offset_loss = row_id * stride_row_loss
    row_offset_lse = row_id * stride_row_lse
    row_offset_zloss = row_id * stride_row_zloss
    col_start = col_block_id * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)

    # Load label
    label = tl.load(labels_ptr + row_id * stride_labels)
    is_ignored = label == ignored_index

    # Initialize partial max for stable softmax
    # Each thread loads one logit
    mask = offs < total_classes
    logits = tl.where(mask, tl.load(logits_ptr + row_offset_logits + offs * stride_col_logits), float("-inf"))
    scaled_logits = logits * logit_scale
    # Use a warp-level reduction for max
    block_max = tl.max(scaled_logits, axis=0)

    # Compute exp(logits - max), then sum
    exp_shifted = tl.exp(scaled_logits - block_max)
    block_sum = tl.sum(exp_shifted, axis=0)

    # For the correct index, compute scaled logit
    correct_logit = tl.where(offs == (label - class_start_idx), scaled_logits, 0.0)
    correct_logit_sum = tl.sum(correct_logit, axis=0)

    # Store partial results in global memory (for multi-block merges if SPLIT)
    # We place partial sums and partial maxima at the row/block index.
    # If we wanted a multi-block reduction, we'd store to a temporary buffer.
    # For simplicity here, do local compute for each block, handle final on block_id=0
    # This store could be replaced by atomic or separate scratch for partial reduction.
    # We'll store partial sums in lse, partial "correct_logit_sum" in z_loss just as placeholders.
    partial_lse = block_max + tl.log(block_sum + 1e-7)
    tl.store(lse_ptr + row_offset_lse + col_block_id, partial_lse)
    tl.store(z_loss_ptr + row_offset_zloss + col_block_id, correct_logit_sum)

    # If this is the first block, finalize the row's cross-entropy using all partial data if not SPLIT.
    if (col_block_id == 0) and (not SPLIT):
        # Accumulate partial sums from all blocks
        partial_lse_vals = [tl.load(lse_ptr + row_offset_lse + b) for b in range((total_classes + BLOCK_SIZE - 1) // BLOCK_SIZE)]
        partial_correct_vals = [tl.load(z_loss_ptr + row_offset_zloss + b) for b in range((total_classes + BLOCK_SIZE - 1) // BLOCK_SIZE)]
        # We reduce them properly
        final_max = partial_lse_vals[0]
        for val in partial_lse_vals:
            final_max = tl.maximum(final_max, val)
        exps = [tl.exp(val - final_max) for val in partial_lse_vals]
        lse_final = final_max + tl.log(tl.sum(exps, 0) + 1e-7)
        correct_sum = tl.sum(partial_correct_vals, 0)

        # Cross entropy loss = - (correct_logit - log_sum_exp)
        nll = -(correct_sum - lse_final)
        # Optional label smoothing
        if HAS_SMOOTHING:
            nll = nll * (1 - smoothing) + lse_final * smoothing

        # If we have a z-loss (for large logit regularization)
        z_loss_val = 0.0
        if lse_square_scale != 0.0:
            z_loss_val = lse_final * lse_final * lse_square_scale

        # If ignoring index, set everything to zero
        if is_ignored:
            nll = 0.0
            z_loss_val = 0.0

        tl.store(loss_ptr + row_offset_loss, nll)
        tl.store(lse_ptr + row_offset_lse, lse_final)
        tl.store(z_loss_ptr + row_offset_zloss, z_loss_val)


@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr,
    labels_ptr,
    lse_ptr,
    dlogits_ptr,
    smoothing,
    logit_scale,
    ignored_index,
    total_classes,
    class_start_idx,
    stride_row_logits,
    stride_col_logits,
    stride_labels,
    stride_row_lse,
    stride_row_dlogits,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr
):
    row_id = tl.program_id(0)
    col_block_id = tl.program_id(1)

    row_offset_logits = row_id * stride_row_logits
    row_offset_lse = row_id * stride_row_lse
    row_offset_dlogits = row_id * stride_row_dlogits
    col_start = col_block_id * BLOCK_SIZE
    offs = col_start + tl.arange(0, BLOCK_SIZE)

    # Load label and lse
    label = tl.load(labels_ptr + row_id * stride_labels)
    is_ignored = label == ignored_index
    # For simplicity, assume the final lse is stored at index 0
    lse_val = tl.load(lse_ptr + row_offset_lse)

    # Read logits, compute probability
    mask = offs < total_classes
    logits = tl.where(mask, tl.load(logits_ptr + row_offset_logits + offs * stride_col_logits), float("-inf"))
    scaled_logits = logits * logit_scale
    exp_val = tl.exp(scaled_logits - lse_val)

    # If label smoothing, each grad is (p - smoothing/total_classes)
    grad_val = exp_val
    if HAS_SMOOTHING:
        grad_val = grad_val - (smoothing / total_classes)

    # For correct label index, subtract 1 - smoothing
    correct_mask = offs == (label - class_start_idx)
    grad_val = tl.where(correct_mask, grad_val - (1.0 - smoothing), grad_val)

    # If ignoring index, set grad to 0
    grad_val = tl.where(is_ignored, 0.0, grad_val)

    # Store
    tl.store(dlog
