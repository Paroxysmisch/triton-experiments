import torch
import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr, lse_ptr, z_loss_ptr,
    total_classes, class_start_idx, logit_scale, smoothing, ignored_index,
    logits_row_stride, logits_col_stride,
    HAS_SMOOTHING: tl.constexpr, SPLIT: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    label = tl.load(labels_ptr + row_idx)
    if label == ignored_index:
        tl.store(loss_ptr + row_idx, 0.0)
        tl.store(lse_ptr + row_idx, 0.0)
        if z_loss_ptr != 0:
            tl.store(z_loss_ptr + row_idx, 0.0)
        return

    max_logit = -tl.inf
    sum_exp = 0.0
    sum_logits = 0.0
    logit_label = 0.0
    start_col = class_start_idx
    end_col = class_start_idx + total_classes

    for col_block_start in range(start_col, end_col, BLOCK_SIZE):
        col_offsets = col_block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < end_col
        logits_offset = row_idx * logits_row_stride + col_offsets * logits_col_stride
        current_logits = tl.load(logits_ptr + logits_offset, mask=mask, other=0.0) * logit_scale
        current_max = tl.max(current_logits, axis=0)
        new_max = tl.maximum(max_logit, current_max)
        if new_max != max_logit:
            sum_exp *= tl.exp(max_logit - new_max)
            max_logit = new_max
        exp_logits = tl.exp(current_logits - max_logit)
        exp_logits = tl.where(mask, exp_logits, 0.0)
        sum_exp += tl.sum(exp_logits, axis=0)
        if HAS_SMOOTHING:
            sum_logits += tl.sum(current_logits, axis=0)
        if (col_block_start <= label) and (label < (col_block_start + BLOCK_SIZE)):
            label_pos = label - col_block_start
            logit_label = tl.load(logits_ptr + row_idx * logits_row_stride + (col_block_start + label_pos) * logits_col_stride) * logit_scale

    lse = tl.log(sum_exp) + max_logit
    if HAS_SMOOTHING:
        avg_logit = sum_logits / total_classes
        loss = (1.0 - smoothing) * (lse - logit_label) + smoothing * (lse - avg_logit)
    else:
        loss = lse - logit_label

    if z_loss_ptr != 0:
        z_loss = 0.5 * (lse ** 2)
        tl.store(z_loss_ptr + row_idx, z_loss)
    tl.store(loss_ptr + row_idx, loss)
    tl.store(lse_ptr + row_idx, lse)

@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr, logits_ptr, labels_ptr, lse_ptr,
    total_classes, class_start_idx, logit_scale, smoothing, ignored_index,
    dlogits_row_stride, dlogits_col_stride,
    logits_row_stride, logits_col_stride,
    HAS_SMOOTHING: tl.constexpr, SPLIT: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    col_start = class_start_idx + col_block_idx * BLOCK_SIZE
    col_offs = col_start + tl.arange(0, BLOCK_SIZE)
    mask = (col_offs < (class_start_idx + total_classes))
    label = tl.load(labels_ptr + row_idx)

    if label == ignored_index:
        dlogits = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    else:
        lse = tl.load(lse_ptr + row_idx)
        logits_offset = row_idx * logits_row_stride + col_offs * logits_col_stride
        logits = tl.load(logits_ptr + logits_offset, mask=mask, other=0.0)
        scaled_logits = logits * logit_scale
        probs = tl.exp(scaled_logits - lse)
        if HAS_SMOOTHING:
            target = tl.full((BLOCK_SIZE,), smoothing / total_classes, dtype=tl.float32)
            target = tl.where(col_offs == label, target + (1.0 - smoothing), target)
        else:
            target = tl.where(col_offs == label, 1.0, 0.0)
        dlogits = (probs - target) * logit_scale
        dlogits = tl.where(mask, dlogits, 0.0)
    dlogits_offset = row_idx * dlogits_row_stride + col_offs * dlogits_col_stride
    tl.store(dlogits_ptr + dlogits_offset, dlogits, mask=mask)

def cross_entropy_fwd(logits, labels, smoothing=0.0, logit_scale=1.0, ignored_index=-100, z_loss=False):
    assert logits.dim() == 2 and labels.dim() == 1
    B, C = logits.shape
    device = logits.device
    loss = torch.empty(B, dtype=torch.float32, device=device)
    lse = torch.empty(B, dtype=torch.float32, device=device)
    z_loss_tensor = torch.empty(B, dtype=torch.float32, device=device) if z_loss else None
    BLOCK_SIZE = 128
    grid = (B,)
    cross_entropy_fwd_kernel[grid](
        logits, labels, loss, lse, z_loss_tensor if z_loss else 0,
        C, 0, logit_scale, smoothing, ignored_index,
        logits.stride(0), logits.stride(1),
        HAS_SMOOTHING=smoothing > 0, SPLIT=False, BLOCK_SIZE=BLOCK_SIZE
    )
    print("Forward pass - Loss:", loss)
    print("Forward pass - LSE:", lse)
    if z_loss:
        print("Forward pass - Z_loss:", z_loss_tensor)
    return loss, lse, z_loss_tensor

def cross_entropy_bwd(logits, labels, lse, smoothing=0.0, logit_scale=1.0, ignored_index=-100):
    B, C = logits.shape
    device = logits.device
    dlogits = torch.empty_like(logits)
    BLOCK_SIZE = 128
    grid = (B, (C + BLOCK_SIZE - 1) // BLOCK_SIZE)
    cross_entropy_bwd_kernel[grid](
        dlogits, logits, labels, lse,
        C, 0, logit_scale, smoothing, ignored_index,
        dlogits.stride(0), dlogits.stride(1),
        logits.stride(0), logits.stride(1),
        HAS_SMOOTHING=smoothing > 0, SPLIT=False, BLOCK_SIZE=BLOCK_SIZE
    )
    print("Backward pass - dlogits:", dlogits)
    return dlogits

# Example usage
torch.manual_seed(0)
B, C = 2, 4
logits = torch.randn(B, C, device='cuda', dtype=torch.float32)
labels = torch.randint(0, C, (B,), device='cuda', dtype=torch.long)
loss, lse, z_loss = cross_entropy_fwd(logits, labels, smoothing=0.1, z_loss=True)
dlogits = cross_entropy_bwd(logits, labels, lse, smoothing=0.1)
