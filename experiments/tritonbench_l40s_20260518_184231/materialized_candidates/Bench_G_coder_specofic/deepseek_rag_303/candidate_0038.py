import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, logits_row_stride, logits_col_stride,
    logits_ptr2, logits_row_stride2, logits_col_stride2,
    labels_ptr,
    loss_ptr, lse_ptr, 
    z_loss_ptr,
    logit_scale_ptr,
    ignored_index,
    total_classes, class_start_idx,
    smoothing,
    logit_scale, lse_square_scale,
    BLOCK_SIZE : tl.constexpr,
    HAS_SMOOTHING : tl.constexpr,
    SPLIT : tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    label_idx_ptr = labels_ptr + row_idx
    label_idx = tl.load(label_idx_ptr)
    if label_idx == ignored_index:
        loss = 0.0
        z_loss = 0.0
        lse = 0.0
        tl.store(loss_ptr + row_idx, loss)
        tl.store(z_loss_ptr + row_idx, z_loss)
        tl.store(lse_ptr + row_idx, lse)
        return
    pass

    col_offsets_mask = col_offsets < total_classes
    if SPLIT :
        col_offsets += class_start_idx
    pass
    logits_ptr_row = logits_ptr + row_idx * logits_row_stride + col_offsets * logits_col_stride
    logits_ptr_row2 = logits_ptr2 + row_idx * logits_row_stride2 + col_offsets * logits_col_stride2
    logits = tl.load(logits_ptr_row, mask = col_offsets_mask, other = 0.0)
    logits2 = tl.load(logits_ptr_row2, mask = col_offsets_mask, other = 0.0)
    logits += logits2

    logits = logits * logit_scale
    orig_logits = logits
    if HAS_SMOOTHING :
        logits = (1 - smoothing) * logits + smoothing * logit_scale
    pass
    lse_unscaled = tl.log(tl.sum(tl.exp(logits), axis = 0))
    lse = lse_unscaled * logit_scale
    z_loss = 0.0 if not HAS_SMOOTHING else lse_square_scale * lse * lse
    if label_idx != ignored_index:
        loss = -orig_logits[label_idx] + z_loss + lse
        tl.store(loss_ptr + row_idx, loss)
    else:
        loss = 0.0
        z_loss = 0.0
    pass
    tl.store(lse_ptr + row_idx, lse)
    tl.store(z_loss_ptr + row_idx, z_loss)
    logit_scale = tl.load(logit_scale_ptr)
    tl.debug_barrier()
    logits = logits * (1.0 / logit_scale)
    lse_unscaled = lse_unscaled * (1.0 / logit_scale)

pass

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, logits_row_stride, logits_col_stride,
    logits_ptr2, logits_row_stride2, logits_col_stride2,
    labels_ptr,
    dlogits_ptr, dlogits2_ptr,
    lse_ptr,
    logit_scale_ptr,
    ignored_index,
    total_classes, class_start_idx,
    smoothing,    
    BLOCK_SIZE : tl.constexpr,
    HAS_SMOOTHING : tl.constexpr,
    SPLIT : tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    label_idx_ptr = labels_ptr + row_idx
    label_idx = tl.load(label_idx_ptr)

    if label_idx == ignored_index: return
    pass
    col_offsets_mask = col_offsets < total_classes
    if SPLIT :
        col_offsets += class_start_idx
    pass
    logits_ptr_row = logits_ptr  + row_idx * logits_row_stride  + col_offsets * logits_col_stride
    logits_ptr_row2 = logits_ptr2  + row_idx * logits_row_stride2 + col_offsets * logits_col_stride2    
    logits = tl.load(logits_ptr_row, mask = col_offsets_mask, other = -float("inf"))
    logits2 = tl.load(logits_ptr_row2, mask = col_offsets_mask, other = -float("inf"))
    logits += logits2

    logit_scale = tl.load(logit_scale_ptr)
    logits = logits * (1.0 / logit_scale)
    lse = tl.load(lse_ptr + row_idx)
    lse = lse * (1.0 / logit_scale)
    probs_hat = tl.exp(logits - lse)

    if HAS_SMOOTHING :
        probs_hat = ((1 - smoothing) * probs_hat) + (smoothing / total_classes)
    pass
    if label_idx != ignored_index:
        probs_hat_ptr = probs_hat_ptr = probs_hat_ptr + (row_idx * col_offsets )
        tl.store(probs_hat_ptr, -probs_hat)
        probs_hat = probs_hat + 1.0
    else:
        probs_hat = probs_hat
    pass
    dlogits = probs_hat
    logits = (probs_hat - 1.0) * logit_scale
    if label_idx != ignored_index:
        logits = logits - lse
    pass
    dlogits_ptr_row = dlogits_ptr + row_idx * logits_row_stride + col_offsets * logits_col_stride
    dlogits2_ptr_row = dlogits2_ptr + row_idx * logits_row_stride2 + col_offsets * logits_col_stride2
    tl.store(dlogits_ptr_row, dlogits, mask = col_offsets_mask)
    tl.store(dlogits2_ptr_row, -dlogits, mask = col_offsets_mask)
pass

import torch
def cross_entropy_fwd(
    logits,
    logits2,
    labels,
    smoothing,
    logit_scale,
    lse_square_scale = 0.0,
    ignored_index = -100,
    total_classes = None,
    class_start_idx = None,
    print_dbg = False,
    logit_scale_ptr = None,
):
    logits = logits.contiguous()
    logits2 = logits2.contiguous()
    batch_size, total_classes = logits.shape
    assert(logits2.shape == logits.shape)
    assert(labels.shape == (batch_size,))
    if total_classes is None:
        total_classes = logits.shape[1]
    if class_start_idx is None:
        class_start_idx = 0
