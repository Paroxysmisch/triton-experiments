import torch
import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr, lse_ptr,
    stride_logits, stride_labels,
    N, C, smoothing, logit_scale,
    BLOCK_SIZE: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    Z_LOSS_FACTOR: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    label = tl.load(labels_ptr + row * stride_labels)
    if label == IGNORE_INDEX:
        tl.store(loss_ptr + row, 0.0)
        tl.store(lse_ptr + row, 0.0)
        return

    max_logit = -tl.float32(float('inf'))
    sum_exp = tl.float32(0.0)
    sum_scaled = tl.float32(0.0)
    scaled_label = tl.float32(0.0)
    label_valid = tl.int32(0)

    for class_chunk in range(0, C, BLOCK_SIZE):
        offs = class_chunk + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        logits = tl.load(logits_ptr + row * stride_logits + offs, mask=mask, other=0)
        scaled = logits * logit_scale

        chunk_max = tl.max(tl.where(mask, scaled, -tl.float32(float('inf'))), axis=0)
        new_max = tl.maximum(max_logit, chunk_max)
        delta = new_max - max_logit
        sum_exp = sum_exp * tl.exp(-delta) + tl.sum(tl.where(mask, tl.exp(scaled - new_max), 0.0), axis=0)
        max_logit = new_max

        sum_scaled += tl.sum(tl.where(mask, scaled, 0.0), axis=0)

        chunk_start = class_chunk
        chunk_end = tl.minimum(class_chunk + BLOCK_SIZE, C)
        if (label >= chunk_start) & (label < chunk_end):
            label_off = label - chunk_start
            label_mask = (tl.arange(0, BLOCK_SIZE) == label_off) & mask
            scaled_label_val = tl.sum(scaled * tl.where(label_mask, 1.0, 0.0), axis=0)
            scaled_label += scaled_label_val
            label_valid = 1

    lse = tl.log(sum_exp + 1e-12) + max_logit
    tl.store(lse_ptr + row, lse)

    if label < 0 or label >= C:
        loss = 0.0
    else:
        C_f = tl.float32(C)
        if label_valid == 1:
            term1 = (1.0 - smoothing) * scaled_label
            term2 = (smoothing / (C_f - 1.0)) * (sum_scaled - scaled_label)
        else:
            term1 = 0.0
            term2 = (smoothing / (C_f - 1.0)) * sum_scaled
        loss = lse - (term1 + term2)
        if Z_LOSS_FACTOR != 0.0:
            loss += Z_LOSS_FACTOR * (lse ** 2)

    tl.store(loss_ptr + row, loss)

@triton.jit
def cross_entropy_bwd_kernel(
    grad_logits_ptr, logits_ptr, lse_ptr, labels_ptr,
    stride_grad, stride_logits, stride_labels,
    N, C, smoothing, logit_scale,
    BLOCK_SIZE: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    Z_LOSS_FACTOR: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // (C // BLOCK_SIZE)
    chunk = (pid % (C // BLOCK_SIZE)) * BLOCK_SIZE

    if row >= N:
        return

    label = tl.load(labels_ptr + row * stride_labels)
    if label == IGNORE_INDEX:
        return

    lse = tl.load(lse_ptr + row)
    C_f = tl.float32(C)

    for class_chunk in range(chunk, chunk + BLOCK_SIZE):
        if class_chunk >= C:
            break
        logit = tl.load(logits_ptr + row * stride_logits + class_chunk)
        scaled_logit = logit * logit_scale
        prob = tl.exp(scaled_logit - lse)
        target = tl.where(class_chunk == label, (1.0 - smoothing), smoothing / (C_f - 1.0))
        grad = (prob - target) * logit_scale

        if Z_LOSS_FACTOR != 0.0:
            z_grad = 2.0 * Z_LOSS_FACTOR * lse * prob * logit_scale
            grad += z_grad

        tl.store(grad_logits_ptr + row * stride_grad + class_chunk, grad)

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing, logit_scale, z_loss_factor, ignore_index=-100):
        N, C = logits.shape
        losses = torch.empty_like(logits[:, 0])
        lse = torch.empty_like(logits[:, 0])

        BLOCK_SIZE = 128
        grid = (N,)
        cross_entropy_fwd_kernel[grid](
            logits, labels, losses, lse,
            logits.stride(0), labels.stride(0),
            N, C, smoothing, logit_scale,
            BLOCK_SIZE=BLOCK_SIZE,
            IGNORE_INDEX=ignore_index,
            Z_LOSS_FACTOR=z_loss_factor,
        )

        ctx.save_for_backward(logits, lse, labels)
        ctx.smoothing = smoothing
        ctx.logit_scale = logit_scale
        ctx.z_loss_factor = z_loss_factor
        ctx.ignore_index = ignore_index

        return losses, lse

    @staticmethod
    def backward(ctx, grad_losses, grad_lse):
        logits, lse, labels = ctx.saved_tensors
        N, C = logits.shape
        grad_logits = torch.zeros_like(logits)

        BLOCK_SIZE = 128
        grid = (N * (C + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        cross_entropy_bwd_kernel[grid](
            grad_logits, logits, lse, labels,
            grad_logits.stride(0), logits.stride(0), labels.stride(0),
            N, C, ctx.smoothing, ctx.logit_scale,
            BLOCK_SIZE=BLOCK_SIZE,
            IGNORE_INDEX=ctx.ignore_index,
            Z_LOSS_FACTOR=ctx.z_loss_factor,
        )

        if grad_losses is not None:
            grad_logits *= grad_losses.unsqueeze(-1)

        return grad_logits, None, None, None, None, None

def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.0,
    logit_scale: float = 1.0,
    z_loss_factor: float = 0.0,
    ignore_index: int = -100,
) -> (torch.Tensor, torch.Tensor):
    losses, lse = CrossEntropyLoss.apply(logits, labels, smoothing, logit_scale, z_loss_factor, ignore_index)
    return losses, lse
