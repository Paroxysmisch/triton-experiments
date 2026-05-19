import torch
import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    loss_ptr,  # data ptrs
    lse_ptr,
    z_loss_ptr,
    logits_ptr,
    labels_ptr,
    smoothing,
    logit_scale,
    lse_square_scale,
    ignored_index,
    total_classes,
    class_start_idx,  # Useful for tensor parallel when each rank only has a subset of classes
    n_cols,  # shapes
    n_rows,
    logits_row_stride,  # strides
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(
        tl.float32
    ) * logit_scale
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    if label_idx == ignored_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(
            n_cols, (col_block_idx + 1) * BLOCK_SIZE
        ):
            logits_label = tl.load(logits_ptr + label_idx) * logit_scale
            if HAS_SMOOTHING:
                loss = (
                    (lse if not SPLIT else 0.0)
                    - smoothing * sum_logits / total_classes
                    - (1 - smoothing) * logits_label
                )
            else:
                loss = (lse if not SPLIT else 0.0) - logits_label
        else:
            if HAS_SMOOTHING:
                loss = smoothing * ((lse if not SPLIT else 0.0) - sum_logits / total_classes)
            else:
                loss = 0.0
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)


@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr,  # data ptrs
    dloss_ptr,
    logits_ptr,
    lse_ptr,
    labels_ptr,
    smoothing,
    logit_scale,
    lse_square_scale,
    ignored_index,
    total_classes,
    class_start_idx,  # Useful for tensor parallel when each rank only has a subset of classes
    n_cols,  # shapes
    logits_row_stride,  # strides
    dlogits_row_stride,
    dloss_row_stride,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != ignored_index:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(
        tl.float32
    ) * logit_scale
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx -= class_start_idx
    if HAS_SMOOTHING:
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 - smoothing), probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, (dloss * logit_scale) * probs, mask=col_offsets < n_cols)

class CrossEntropyLoss:
    def __init__(self, smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0, ignored_index=-100, process_group=None):
        self.smoothing = smoothing
        self.logit_scale = logit_scale
        self.lse_square_scale = lse_square_scale
        self.ignored_index = ignored_index
        self.process_group = process_group

    def forward(self, logits, labels, total_classes, class_start_idx, n_cols, n_rows, logits_row_stride, BLOCK_SIZE, SPLIT=False):
        loss = torch.empty_like(logits)
        lse = torch.empty_like(logits)
        z_loss = torch.empty_like(logits) if not SPLIT else None

        grid = (n_rows, (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE)
        cross_entropy_fwd_kernel[grid](
            loss, lse, z_loss, logits, labels, self.smoothing, self.logit_scale, self.lse_square_scale,
            self.ignored_index, total_classes, class_start_idx, n_cols, n_rows, logits_row_stride,
            BLOCK_SIZE, self.smoothing > 0, SPLIT
        )

        return loss, lse, z_loss

    def backward(self, dlogits, dloss, logits, lse, labels, total_classes, class_start_idx, n_cols, logits_row_stride, dlogits_row_stride, dloss_row_stride, BLOCK_SIZE):
        grid = (dlogits.shape[0], (dlogits.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE)
        cross_entropy_bwd_kernel[grid](
            dlogits, dloss, logits, lse, labels, self.smoothing, self.logit_scale, self.lse_square_scale,
            self.ignored_index, total_classes, class_start_idx, n_cols, logits_row_stride,
            dlogits_row_stride, dloss_row_stride, BLOCK_SIZE, self.smoothing > 0
        )


def cross_entropy_loss(logits, labels, total_classes, class_start_idx, smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0, ignored_index=-100, process_group=None, BLOCK_SIZE=128, SPLIT=False):
    ce_loss = CrossEntropyLoss(smoothing, logit_scale, lse_square_scale, ignored_index, process_group)
    n_cols = logits.shape[1]
    n_rows = logits.shape[0]
    logits_row_stride = logits.stride(0)

    # Forward pass
    loss, lse, z_loss = ce_loss.forward(logits, labels, total_classes, class_start_idx, n_cols, n_rows, logits_row_stride, BLOCK_SIZE, SPLIT)

    # Backward pass (if needed)
    dlogits = torch.zeros_like(logits)
    dloss = torch.ones_like(loss)  # Assuming dloss is 1 for simplicity
    dlogits_row_stride = dlogits.stride(0)
    dloss_row_stride = dloss.stride(0)
    ce_loss.backward(dlogits, dloss, logits, lse, labels, total_classes, class_start_idx, n_cols, logits_row_stride, dlogits_row_stride, dloss_row_stride, BLOCK_SIZE)

    return loss, z_loss, dlogits
