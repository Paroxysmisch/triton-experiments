import triton
import triton.language as tl
import torch
from .utils import triton_tanh

MAX_FUSED_SIZE = 1024  # Adjusted to fit GPU thread limits

def calculate_settings(n):
    max_block = min(n, MAX_FUSED_SIZE)
    block_size = 1
    while block_size * 2 <= max_block:
        block_size *= 2
    num_warps = block_size // 32
    return block_size, num_warps

@triton.heuristics({
    "DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def _cross_entropy_forward(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING:
        logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING:
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = logits.to(tl.float32)
    max_logit = tl.max(logits, 0)
    logsumexp = max_logit + tl.log(tl.sum(tl.exp(logits - max_logit), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx)
        if DO_LOGIT_SCALING:
            x = LOGIT_SCALE * x
        if DO_SOFTCAPPING:
            x = SOFTCAP * triton_tanh(x / SOFTCAP)
        loss = logsumexp - x.to(tl.float32)
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)

@triton.heuristics({
    "DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def _chunked_cross_entropy_forward(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx

    col_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING:
        logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING:
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = logits.to(tl.float32)
    max_logit = tl.max(logits, 0)
    partial_logsumexp = max_logit + tl.log(tl.sum(tl.exp(logits - max_logit), 0))

    if chunk_idx == 0:
        loss = 0.0
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            if DO_LOGIT_SCALING:
                x = LOGIT_SCALE * x
            if DO_SOFTCAPPING:
                x = SOFTCAP * triton_tanh(x / SOFTCAP)
            loss = -x
        tl.store(loss_ptr, loss)
    tl.store(logsumexp_ptr, partial_logsumexp)

@triton.heuristics({
    "DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def _cross_entropy_backward(
    logits_ptr, logits_row_stride,
    dloss_ptr, dloss_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * logits_row_stride
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    dloss = tl.load(dloss_ptr) if label_idx != -100 else 0.0

    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    scaled_x = x * LOGIT_SCALE if DO_LOGIT_SCALING else x
    if DO_SOFTCAPPING:
        tanh_x = triton_tanh(scaled_x / SOFTCAP)
        scaled_x = SOFTCAP * tanh_x

    logsumexp = tl.load(logsumexp_ptr + row_idx)
    softmax_grad = tl.exp(scaled_x.to(tl.float32) - logsumexp)
    softmax_grad = tl.where(col_offsets == label_idx, softmax_grad - 1.0, softmax_grad)

    if DO_LOGIT_SCALING:
        softmax_grad *= LOGIT_SCALE
    if DO_SOFTCAPPING:
        softmax_grad *= (1.0 - tanh_x * tanh_x)

    tl.store(logits_ptr + col_offsets, dloss * softmax_grad, mask=mask)

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_softcapping=0, logit_scaling=0):
        n_rows, vocab_size = logits.shape
        ctx.save_for_backward(logits, labels)
        ctx.DO_SOFTCAPPING = logit_softcapping != 0
        ctx.logit_softcapping = logit_softcapping
        ctx.DO_LOGIT_SCALING = logit_scaling != 0
        ctx.logit_scaling = logit_scaling

        n_chunks = (vocab_size + MAX_FUSED_SIZE - 1) // MAX_FUSED_SIZE
        losses = torch.empty(n_rows, device=logits.device, dtype=torch.float32)

        if n_chunks == 1:
            block_size, num_warps = calculate_settings(vocab_size)
            logsumexp = torch.empty_like(losses)
            _cross_entropy_forward[(n_rows,)](
                logits, logits.stride(0),
                losses, logsumexp, labels,
                VOCAB_SIZE=vocab_size, BLOCK_SIZE=block_size,
                DO_SOFTCAPPING=ctx.DO_SOFTCAPPING, SOFTCAP=logit_softcapping,
                DO_LOGIT_SCALING=ctx.DO_LOGIT_SCALING, LOGIT_SCALE=logit_scaling,
                num_warps=num_warps
            )
        else:
            logsumexp = torch.empty((n_rows, n_chunks), device=logits.device, dtype=torch.float32)
            _chunked_cross_entropy_forward[(n_rows, n_chunks)](
                logits, logits.stride(0),
                losses, logsumexp, labels,
                VOCAB_SIZE=vocab_size, N_CHUNKS=n_chunks,
                BLOCK_SIZE=MAX_FUSED_SIZE,
                DO_SOFTCAPPING=ctx.DO_SOFTCAPPING, SOFTCAP=logit_softcapping,
                DO_LOGIT_SCALING=ctx.DO_LOGIT_SCALING, LOGIT_SCALE=logit_scaling,
                num_warps=32
            )
            logsumexp = torch.logsumexp(logsumexp, dim=1)
            losses += logsumexp
            losses[labels == -100] = 0.0

        ctx.save_for_backward(logits, labels, logsumexp)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, logsumexp = ctx.saved_tensors
        n_rows, vocab_size = logits.shape

        grad_logits = torch.zeros_like(logits)
        BLOCK_SIZE = 1024
        n_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE

        _cross_entropy_backward[(n_rows, n_blocks)](
            logits, logits.stride(0),
            grad_output, grad_output.stride(0),
            logsumexp, labels,
            VOCAB_SIZE=vocab_size, BLOCK_SIZE=BLOCK_SIZE,
            DO_SOFTCAPPING=ctx.DO_SOFTCAPPING, SOFTCAP=ctx.logit_softcapping,
            DO_LOGIT_SCALING=ctx.DO_LOGIT_SCALING, LOGIT_SCALE=ctx.logit_scaling,
            num_warps=8
        )
        return grad_logits, None, None, None

def fast_cross_entropy_loss(logits, labels, logit_softcapping=0, logit_scaling=0, n_items=None):
    batch, seq_len, d = logits.shape
    labels = labels.view(-1)
    logits = logits.view(batch * seq_len, d)
    loss = Fast_CrossEntropyLoss.apply(logits, labels, logit_softcapping, logit_scaling)
    if n_items is None:
        n_items = (labels != -100).sum()
    return loss.sum() / n_items.clamp(min=1)
