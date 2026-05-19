import triton
import triton.language as tl
import torch

@triton.heuristics({
    "HAS_SMOOTHING": lambda args: args["HAS_SMOOTHING"],
    "SPLIT": lambda args: args["SPLIT"],
})
@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, logits_row_stride,
    loss_ptr, lse_ptr, z_loss_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SMOOTHING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
    LSE_SQUARE_SCALE: tl.constexpr,
    IGNORED_INDEX: tl.constexpr,
    SPLIT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    lse_ptr += row_idx * SPLIT + block_idx
    z_loss_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if LOGIT_SCALE != 1.0: logits = LOGIT_SCALE * logits

    logits = logits.to(tl.float32)
    c = tl.max(logits, 0)
    lse = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if block_idx == 0:
        if label_idx != IGNORED_INDEX:
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            if LOGIT_SCALE != 1.0: x = LOGIT_SCALE * x
            loss = lse - x
            if HAS_SMOOTHING:
                loss = (1 - SMOOTHING) * loss + SMOOTHING * lse
            if LSE_SQUARE_SCALE != 0.0:
                z_loss = LSE_SQUARE_SCALE * (lse ** 2)
            else:
                z_loss = 0.0
        else:
            loss = 0.0
            z_loss = 0.0
        tl.store(loss_ptr, loss)
        tl.store(z_loss_ptr, z_loss)
    tl.store(lse_ptr, lse)

@triton.heuristics({
    "HAS_SMOOTHING": lambda args: args["HAS_SMOOTHING"],
})
@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, logits_row_stride,
    dlogits_ptr, dlogits_row_stride,
    lse_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SMOOTHING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
    IGNORED_INDEX: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr += row_idx * dlogits_row_stride
    lse_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)

    if label_idx != IGNORED_INDEX:
        dloss = 1.0
    else:
        dloss = 0.0

    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    if LOGIT_SCALE != 1.0: logits = LOGIT_SCALE * logits

    lse = tl.load(lse_ptr).to(tl.float32)
    y = tl.exp(logits - lse)
    y = tl.where(col_offsets == label_idx, y - 1.0, y)

    if HAS_SMOOTHING:
        y = (1 - SMOOTHING) * y + SMOOTHING

    if LOGIT_SCALE != 1.0: y = y * LOGIT_SCALE

    tl.store(dlogits_ptr + col_offsets, dloss * y, mask=mask)

import torch
from torch.autograd import Function

MAX_FUSED_SIZE = 65536

class Fast_CrossEntropyLoss(Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0, ignored_index=-100):
        n_rows, vocab_size = logits.shape

        div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
        n_chunks = div + (mod != 0)
        losses = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")
        z_losses = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")

        HAS_SMOOTHING = (smoothing != 0.0)
        SPLIT = (n_chunks > 1)

        if n_chunks == 1:
            BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
            lse = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")

            cross_entropy_fwd_kernel[(n_rows,)](
                logits, logits.stride(0),
                losses,
                lse,
                z_losses,
                labels,
                VOCAB_SIZE=vocab_size,
                BLOCK_SIZE=BLOCK_SIZE,
                HAS_SMOOTHING=HAS_SMOOTHING,
                SMOOTHING=smoothing,
                LOGIT_SCALE=logit_scale,
                LSE_SQUARE_SCALE=lse_square_scale,
                IGNORED_INDEX=ignored_index,
                SPLIT=SPLIT,
                num_warps=num_warps,
            )
        else:
            lse = torch.empty((n_rows, n_chunks), dtype=torch.float32, device="cuda:0")

            cross_entropy_fwd_kernel[(n_rows, n_chunks)](
                logits, logits.stride(0),
                losses,
                lse,
                z_losses,
                labels,
                VOCAB_SIZE=vocab_size,
                BLOCK_SIZE=MAX_FUSED_SIZE,
                HAS_SMOOTHING=HAS_SMOOTHING,
                SMOOTHING=smoothing,
                LOGIT_SCALE=logit_scale,
                LSE_SQUARE_SCALE=lse_square_scale,
                IGNORED_INDEX=ignored_index,
                SPLIT=SPLIT,
                num_warps=32,
            )
            lse = torch.logsumexp(lse, dim=1)
            losses += lse
            losses.masked_fill_(labels == ignored_index, 0)
            z_losses.masked_fill_(labels == ignored_index, 0)

        ctx.save_for_backward(logits, lse, labels)
        ctx.HAS_SMOOTHING = HAS_SMOOTHING
        ctx.smoothing = smoothing
        ctx.logit_scale = logit_scale
        ctx.ignored_index = ignored_index
        return losses, lse, z_losses

    @staticmethod
    def backward(ctx, grad_losses, grad_lse, grad_z_losses):
        logits, lse, labels = ctx.saved_tensors
        n_rows, vocab_size = logits.shape

        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)

        dlogits = torch.zeros_like(logits)

        cross_entropy_bwd_kernel[(n_rows, n_blocks)](
            logits, logits.stride(0),
            dlogits, dlogits.stride(0),
            lse,
            labels,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_SMOOTHING=ctx.HAS_SMOOTHING,
            SMOOTHING=ctx.smoothing,
            LOGIT_SCALE=ctx.logit_scale,
            IGNORED_INDEX=ctx.ignored_index,
            num_warps=8,
        )

        return dlogits, None, None, None, None, None

def fast_cross_entropy_loss(
    logits,
    labels,
    smoothing=0.0,
    logit_scale=1.0,
    lse_square_scale=0.0,
    ignored_index=-100,
    n_items=None,
):
    batch, seq_len, d = logits.shape
    assert(labels.shape == (batch, seq_len))

    losses, lse, z_losses = Fast_CrossEntropyLoss.apply(
        logits.view(batch * seq_len, d),
        labels.view(-1),
        smoothing,
        logit_scale,
        lse_square_scale,
        ignored_index,
    )
    if n_items is None:
        n_items = torch.count_nonzero(labels != ignored_index)
    return losses.sum() / n_items, lse, z_losses
