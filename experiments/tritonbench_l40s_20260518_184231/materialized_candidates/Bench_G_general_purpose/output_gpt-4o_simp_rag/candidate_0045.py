import triton
import triton.language as tl
import torch

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, logits_stride,
    loss_ptr, lse_ptr, z_loss_ptr,
    labels_ptr,
    smoothing, logit_scale, lse_square_scale,
    ignored_index, total_classes, class_start_idx,
    BLOCK_SIZE: tl.constexpr, HAS_SMOOTHING: tl.constexpr, SPLIT: tl.constexpr
):
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_stride
    loss_ptr += row_idx
    lse_ptr += row_idx
    z_loss_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < total_classes

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if logit_scale != 1.0:
        logits *= logit_scale

    max_logit = tl.max(logits, 0)
    logsumexp = max_logit + tl.log(tl.sum(tl.exp(logits - max_logit), 0))

    if label_idx != ignored_index:
        true_logit = tl.load(logits_ptr + label_idx)
        if logit_scale != 1.0:
            true_logit *= logit_scale
        loss = logsumexp - true_logit
    else:
        loss = 0.0

    if HAS_SMOOTHING:
        loss = (1 - smoothing) * loss + smoothing * logsumexp

    tl.store(lse_ptr, logsumexp)
    tl.store(loss_ptr, loss)

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, logits_stride,
    dloss_ptr, lse_ptr,
    dlogits_ptr,
    labels_ptr,
    smoothing, logit_scale, lse_square_scale,
    ignored_index, total_classes, class_start_idx,
    BLOCK_SIZE: tl.constexpr, HAS_SMOOTHING: tl.constexpr, SPLIT: tl.constexpr
):
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_stride
    dlogits_ptr += row_idx * logits_stride
    dloss_ptr += row_idx
    lse_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < total_classes
    label_idx = tl.load(labels_ptr).to(tl.int32)

    dloss = tl.load(dloss_ptr)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if logit_scale != 1.0:
        logits *= logit_scale

    logsumexp = tl.load(lse_ptr)
    probs = tl.exp(logits - logsumexp)

    if label_idx != ignored_index:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)

    if HAS_SMOOTHING:
        probs = (1 - smoothing) * probs + smoothing / total_classes

    if logit_scale != 1.0:
        probs *= logit_scale

    tl.store(dlogits_ptr + col_offsets, dloss * probs, mask=mask)

class FastCrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, logit_scale=1.0):
        n_rows, vocab_size = logits.shape
        BLOCK_SIZE = 128  # Example block size
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        z_loss = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        cross_entropy_fwd_kernel[(n_rows,)](
            logits, logits.stride(0),
            losses, lse, z_loss,
            labels,
            smoothing, logit_scale, 1.0,  # lse_square_scale is not used here
            -100, vocab_size, 0,
            BLOCK_SIZE=BLOCK_SIZE, HAS_SMOOTHING=(smoothing > 0), SPLIT=False
        )

        ctx.save_for_backward(logits, labels, lse)
        ctx.smoothing = smoothing
        ctx.logit_scale = logit_scale
        return losses

    @staticmethod
    def backward(ctx, dlosses):
        logits, labels, lse = ctx.saved_tensors
        n_rows, vocab_size = logits.shape
        BLOCK_SIZE = 128  # Example block size
        dlogits = torch.empty_like(logits)

        cross_entropy_bwd_kernel[(n_rows,)](
            logits, logits.stride(0),
            dlosses, lse,
            dlogits,
            labels,
            ctx.smoothing, ctx.logit_scale, 1.0,
            -100, vocab_size, 0,
            BLOCK_SIZE=BLOCK_SIZE, HAS_SMOOTHING=(ctx.smoothing > 0), SPLIT=False
        )

        return dlogits, None, None, None

def fast_cross_entropy_loss(logits, labels, smoothing=0.0, logit_scale=1.0):
    return FastCrossEntropyLoss.apply(logits, labels, smoothing, logit_scale)
