import triton
import triton.language as tl
import torch

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr, lse_ptr, z_loss_ptr,
    total_classes, class_start_idx, logit_scale, smoothing, lse_square_scale, ignored_index,
    BLOCK_SIZE: tl.constexpr, HAS_SMOOTHING: tl.constexpr, SPLIT: tl.constexpr
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * total_classes + block_idx * BLOCK_SIZE
    labels_ptr += row_idx
    loss_ptr += row_idx
    lse_ptr += row_idx
    z_loss_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < total_classes

    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if logit_scale != 1.0:
        logits = logit_scale * logits

    logits = logits.to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    label_idx = tl.load(labels_ptr).to(tl.int32)
    if label_idx != ignored_index:
        x = tl.load(logits_ptr + label_idx)
        if logit_scale != 1.0:
            x = logit_scale * x
        loss = logsumexp - x.to(tl.float32)
        if HAS_SMOOTHING:
            loss = (1 - smoothing) * loss + smoothing * logsumexp / total_classes
        z_loss = lse_square_scale * logsumexp * logsumexp
    else:
        loss = 0.0
        z_loss = 0.0

    tl.store(loss_ptr, loss)
    tl.store(lse_ptr, logsumexp)
    tl.store(z_loss_ptr, z_loss)

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, labels_ptr, dlogits_ptr, dloss_ptr, lse_ptr,
    total_classes, class_start_idx, logit_scale, smoothing, ignored_index,
    BLOCK_SIZE: tl.constexpr, HAS_SMOOTHING: tl.constexpr
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * total_classes + block_idx * BLOCK_SIZE
    dlogits_ptr += row_idx * total_classes + block_idx * BLOCK_SIZE
    labels_ptr += row_idx
    dloss_ptr += row_idx
    lse_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < total_classes

    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if logit_scale != 1.0:
        logits = logit_scale * logits

    logits = logits.to(tl.float32)
    logsumexp = tl.load(lse_ptr)
    probabilities = tl.exp(logits - logsumexp)

    label_idx = tl.load(labels_ptr).to(tl.int32)
    if label_idx != ignored_index:
        dloss = tl.load(dloss_ptr)
        probabilities = tl.where(col_offsets == label_idx, probabilities - 1.0, probabilities)
        if HAS_SMOOTHING:
            probabilities = (1 - smoothing) * probabilities + smoothing / total_classes
        if logit_scale != 1.0:
            probabilities = logit_scale * probabilities
        gradients = dloss * probabilities
    else:
        gradients = 0.0

    tl.store(dlogits_ptr + col_offsets, gradients, mask=mask)

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_scale=1.0, smoothing=0.0, lse_square_scale=0.0, ignored_index=-100, BLOCK_SIZE=128, HAS_SMOOTHING=False, SPLIT=False):
        n_rows, total_classes = logits.shape

        loss = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        z_loss = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        grid = (n_rows, (total_classes + BLOCK_SIZE - 1) // BLOCK_SIZE)
        cross_entropy_fwd_kernel[grid](
            logits, labels, loss, lse, z_loss,
            total_classes, 0, logit_scale, smoothing, lse_square_scale, ignored_index,
            BLOCK_SIZE=BLOCK_SIZE, HAS_SMOOTHING=HAS_SMOOTHING, SPLIT=SPLIT
        )

        ctx.save_for_backward(logits, labels, lse)
        ctx.logit_scale = logit_scale
        ctx.smoothing = smoothing
        ctx.ignored_index = ignored_index
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.HAS_SMOOTHING = HAS_SMOOTHING

        return loss, lse, z_loss

    @staticmethod
    def backward(ctx, dloss, dlse, dz_loss):
        logits, labels, lse = ctx.saved_tensors
        n_rows, total_classes = logits.shape

        dlogits = torch.zeros_like(logits)

        grid = (n_rows, (total_classes + ctx.BLOCK_SIZE - 1) // ctx.BLOCK_SIZE)
        cross_entropy_bwd_kernel[grid](
            logits, labels, dlogits, dloss, lse,
            total_classes, 0, ctx.logit_scale, ctx.smoothing, ctx.ignored_index,
            BLOCK_SIZE=ctx.BLOCK_SIZE, HAS_SMOOTHING=ctx.HAS_SMOOTHING
        )

        return dlogits, None, None, None, None, None, None, None, None

def cross_entropy_fwd(logits, labels, logit_scale=1.0, smoothing=0.0, lse_square_scale=0.0, ignored_index=-100, BLOCK_SIZE=128, HAS_SMOOTHING=False, SPLIT=False):
    return CrossEntropyLoss.apply(logits, labels, logit_scale, smoothing, lse_square_scale, ignored_index, BLOCK_SIZE, HAS_SMOOTHING, SPLIT)

def cross_entropy_bwd(dloss, logits, labels, lse, logit_scale=1.0, smoothing=0.0, ignored_index=-100, BLOCK_SIZE=128, HAS_SMOOTHING=False):
    dlogits = torch.zeros_like(logits)
    CrossEntropyLoss.backward(dloss, None, None, logits, labels, lse, logit_scale, smoothing, ignored_index, BLOCK_SIZE, HAS_SMOOTHING)
    return dlogits
