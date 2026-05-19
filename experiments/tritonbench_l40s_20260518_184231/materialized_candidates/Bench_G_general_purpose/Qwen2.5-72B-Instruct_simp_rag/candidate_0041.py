import triton
import triton.language as tl
import torch
from .utils import triton_tanh

@triton.heuristics({
    "HAS_SMOOTHING": lambda args: args["HAS_SMOOTHING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
    "SPLIT": lambda args: args["SPLIT"],
})
@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    z_loss_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SMOOTHING: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
    SPLIT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    z_loss_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits

    logits = logits.to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx)
        if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
        loss = logsumexp - x.to(tl.float32)
        if HAS_SMOOTHING:
            loss = (1 - SMOOTHING) * loss - SMOOTHING * (logits - c).mean()
    else:
        loss = 0.0

    z_loss = (logsumexp - c) ** 2

    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)
    tl.store(z_loss_ptr, z_loss)

@triton.heuristics({
    "HAS_SMOOTHING": lambda args: args["HAS_SMOOTHING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, logits_row_stride,
    dloss_ptr, dloss_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SMOOTHING: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING:
        x = x * LOGIT_SCALE

    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x.to(tl.float32) - logsumexp)
    if label_idx != -100:
        y = tl.where(
            col_offsets == label_idx,
            y - 1.0,
            y,
        )
    if HAS_SMOOTHING:
        y = y - SMOOTHING * (1 - (col_offsets == label_idx))

    if DO_LOGIT_SCALING:
        y = y * LOGIT_SCALE

    tl.store(logits_ptr + col_offsets, dloss * y, mask=mask)

import torch

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing=0, logit_scale=1.0, lse_square_scale=1.0, ignored_index=-100):
        n_rows, vocab_size = logits.shape

        losses = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")
        logsumexp = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")
        z_loss = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")

        HAS_SMOOTHING = (smoothing != 0)
        DO_LOGIT_SCALING = (logit_scale != 1.0)

        BLOCK_SIZE = 1024
        num_warps = 4

        cross_entropy_fwd_kernel[(n_rows,)](
            logits, logits.stride(0),
            losses,
            logsumexp,
            z_loss,
            labels,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_SMOOTHING=HAS_SMOOTHING,
            SMOOTHING=smoothing,
            DO_LOGIT_SCALING=DO_LOGIT_SCALING,
            LOGIT_SCALE=logit_scale,
            SPLIT=False,
            num_warps=num_warps,
        )

        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.HAS_SMOOTHING = HAS_SMOOTHING
        ctx.smoothing = smoothing
        ctx.DO_LOGIT_SCALING = DO_LOGIT_SCALING
        ctx.logit_scale = logit_scale
        ctx.lse_square_scale = lse_square_scale
        ctx.ignored_index = ignored_index

        return losses

    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, vocab_size = logits.shape

        BLOCK_SIZE = 1024
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)

        cross_entropy_bwd_kernel[(n_rows, n_blocks,)](
            logits, logits.stride(0),
            dlosses, dlosses.stride(0),
            logsumexp,
            labels,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_SMOOTHING=ctx.HAS_SMOOTHING,
            SMOOTHING=ctx.smoothing,
            DO_LOGIT_SCALING=ctx.DO_LOGIT_SCALING,
            LOGIT_SCALE=ctx.logit_scale,
            num_warps=4,
        )

        return logits, None, None, None, None, None

def fast_cross_entropy_loss(
    logits,
    labels,
    smoothing=0,
    logit_scale=1.0,
    lse_square_scale=1.0,
    ignored_index=-100,
    n_items=None,
):
    batch, seq_len, d = logits.shape
    assert(labels.shape == (batch, seq_len))

    loss = Fast_CrossEntropyLoss.apply(
        logits.view(batch * seq_len, d),
        labels.view(-1),
        smoothing,
        logit_scale,
        lse_square_scale,
        ignored_index,
    )
    if n_items is None:
        n_items = torch.count_nonzero(labels != ignored_index)
    return loss.sum() / n_items
