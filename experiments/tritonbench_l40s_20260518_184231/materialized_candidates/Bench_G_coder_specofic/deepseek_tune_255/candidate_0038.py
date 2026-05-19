import torch
import triton
import triton.language as tl

# Triton kernel for forward pass
@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, logits_row_stride,
    loss_ptr, lse_ptr, z_loss_ptr,
    labels_ptr,
    smoothing, logit_scale, lse_square_scale,
    ignored_index,
    total_classes, class_start_idx,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < total_classes

    label_idx = tl.load(labels_ptr + row_idx)
    is_ignored = label_idx == ignored_index

    if SPLIT:
        logit_offset = class_start_idx
    else:
        logit_offset = 0

    logits = tl.load(logits_ptr + row_idx * logits_row_stride + logit_offset + col_offsets, mask=mask, other=-float("inf"))
    logits *= logit_scale

    max_logits = tl.max(logits, axis=0)
    numerator = tl.exp(logits - max_logits)

    if HAS_SMOOTHING:
        denominator = tl.sum(numerator)
        smooth_pos = denominator / total_classes
        smooth_neg = -numerator / total_classes
        loss = tl.where(col_offsets == label_idx, smooth_pos, smooth_neg)
    else:
        denominator = tl.sum(tl.exp(logits - max_logits), axis=0)
        loss = -(tl.exp(logits[label_idx] - max_logits) - denominator) / denominator

    lse = max_logits + tl.log(denominator)
    if HAS_SMOOTHING:
        z_loss = tl.sum(smooth_pos * (smooth_pos - smooth_neg) + smooth_neg * (smooth_neg - smooth_pos))
    else:
        z_loss = tl.sum(numerator * (numerator - denominator) + denominator * (denominator - numerator))

    z_loss *= lse_square_scale

    if SPLIT:
        tl.store(loss_ptr + row_idx, loss if not is_ignored else 0.0, mask=not is_ignored)
        tl.store(lse_ptr + row_idx, lse if not is_ignored else 0.0, mask=not is_ignored)
        tl.store(z_loss_ptr + row_idx, z_loss if not is_ignored else 0.0, mask=not is_ignored)
    else:
        tl.store(loss_ptr + row_idx * BLOCK_SIZE + col_offsets, loss, mask=mask)
        tl.store(lse_ptr + row_idx * BLOCK_SIZE + col_offsets, lse, mask=mask)
        tl.store(z_loss_ptr + row_idx * BLOCK_SIZE + col_offsets, z_loss, mask=mask)

# Wrapper function for forward pass
def cross_entropy_fwd(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float,
    logit_scale: float,
    lse_square_scale: float,
    ignored_index: int,
    total_classes: int,
    class_start_idx: int,
    BLOCK_SIZE: int,
    split: bool,
):
    assert logits.is_contiguous()
    assert labels.is_contiguous()
    assert logits.ndim == 2
    assert labels.ndim == 1
    assert logits.size(0) == labels.size(0)
    assert torch.all((0 <= labels) & (labels < total_classes)), "Labels should be in [0, total_classes)"

    loss = torch.empty_like(labels, dtype=torch.float32, device=logits.device)
    lse = torch.empty_like(logits, dtype=torch.float32, device=logits.device)
    z_loss = torch.empty_like(logits, dtype=torch.float32, device=logits.device)

    grid = lambda META: (logits.size(0), triton.cdiv(total_classes, META["BLOCK_SIZE"]))

    print("-" * 100)

    cross_entropy_fwd_kernel[grid](
        logits, logits.stride(0),
        loss, lse, z_loss,
        labels,
        smoothing, logit_scale, lse_square_scale,
        ignored_index,
        total_classes, class_start_idx,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_SMOOTHING=smoothing > 0.0,
        SPLIT=split,
    )

    print("-" * 100)
    print(f"loss: {loss}")
    print(f"lse: {lse}")
    print(f"z_loss: {z_loss}")

    return loss, lse, z_loss

# Triton kernel for backward pass
@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, logits_row_stride,
    dlogits_ptr,
    lse_ptr,
    labels_ptr,
    smoothing,
    logit_scale,
    ignored_index,
    total_classes, class_start_idx,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < total_classes

    label_idx = tl.load(labels_ptr + row_idx)
    is_ignored = label_idx == ignored_index

    if SPLIT:
        logit_offset = class_start_idx
    else:
        logit_offset = 0

    logits = tl.load(logits_ptr + row_idx * logits_row_stride + logit_offset + col_offsets, mask=mask, other=-float("inf")) * logit_scale
    lse = tl.load(lse_ptr + row_idx * logits_row_stride + logit_offset + col_offsets, mask=mask)

    if HAS_SMOOTHING:
        smooth_pos = tl.sum(tl.where(col_offsets == label_idx, 1.0, 0.0))
        smooth_neg = -tl.sum(tl.where(col_offsets == label_idx, 1.0, 0.0))
        dlogits = tl.where(col_offsets == label_idx, smooth_pos, smooth_neg)
    else:
        probs = tl.exp(logits - lse)
        dlogits = tl.where(col_offsets == label_idx, 1.0 - tl.sum(probs, axis=0), -tl.sum(probs, axis=0))

    tl.store(dlogits_ptr + row_idx * logits_row_stride + logit_offset + col_offsets, dlogits, mask=mask)

# Wrapper function for backward pass
def cross_entropy_bwd(
    logits: torch.Tensor,
    lse:
