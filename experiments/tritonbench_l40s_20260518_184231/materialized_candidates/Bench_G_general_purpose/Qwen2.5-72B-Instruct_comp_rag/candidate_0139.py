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
    def __init__(self, smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0, ignored_index=-100, total_classes=None, class_start_idx=0, split=False):
        self.smoothing = smoothing
        self.logit_scale = logit_scale
        self.lse_square_scale = lse_square_scale
        self.ignored_index = ignored_index
        self.total_classes = total_classes
        self.class_start_idx = class_start_idx
        self.split = split

    def forward(self, logits, labels):
        n_rows, n_cols = logits.shape
        logits_row_stride = logits.stride(0)
        loss = torch.empty_like(logits)
        lse = torch.empty_like(logits)
        z_loss = torch.empty_like(logits)

        grid = (n_rows, (n_cols + 1024 - 1) // 1024)
        cross_entropy_fwd_kernel[grid](
            loss, lse, z_loss, logits, labels, self.smoothing, self.logit_scale, self.lse_square_scale,
            self.ignored_index, self.total_classes, self.class_start_idx, n_cols, n_rows, logits_row_stride,
            BLOCK_SIZE=1024, HAS_SMOOTHING=self.smoothing > 0, SPLIT=self.split
        )

        if self.split:
            loss = loss.sum(dim=1)
        else:
            loss = loss.sum(dim=1) + z_loss.sum(dim=1)

        return loss, lse

    def backward(self, dloss, logits, lse, labels, dlogits=None, inplace=False):
        n_rows, n_cols = logits.shape
        logits_row_stride = logits.stride(0)
        dlogits_row_stride = dlogits.stride(0) if dlogits is not None else logits_row_stride
        dloss_row_stride = dloss.stride(0)

        if inplace:
            dlogits = logits
        else:
            dlogits = torch.zeros_like(logits)

        grid = (n_rows, (n_cols + 1024 - 1) // 1024)
        cross_entropy_bwd_kernel[grid](
            dlogits, dloss, logits, lse, labels, self.smoothing, self.logit_scale, self.lse_square_scale,
            self.ignored_index, self.total_classes, self.class_start_idx, n_cols,
            logits_row_stride, dlogits_row_stride, dloss_row_stride,
            BLOCK_SIZE=1024, HAS_SMOOTHING=self.smoothing > 0
        )

        return dlogits

def cross_entropy_loss(logits, labels, smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0, ignored_index=-100, total_classes=None, class_start_idx=0, split=False, inplace_backward=False):
    loss_fn = CrossEntropyLoss(smoothing, logit_scale, lse_square_scale, ignored_index, total_classes, class_start_idx, split)
    loss, lse = loss_fn.forward(logits, labels)
    dlogits = loss_fn.backward(dloss=torch.ones_like(loss), logits=logits, lse=lse, labels=labels, inplace=inplace_backward)
    return loss, lse, dlogits
