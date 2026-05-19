import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr,  # pointer to logits tensor
    labels_ptr,  # pointer to labels tensor
    loss_ptr,    # pointer to loss tensor
    lse_ptr,     # pointer to LSE tensor
    smoothing,   # label smoothing factor
    lse_square_scale,  # LSE regularization scale
    ignored_index,  # ignored label index
    N,  # number of samples
    C,  # number of classes
    BLOCK_SIZE: tl.constexpr  # block size
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    logits = tl.load(logits_ptr + offsets[:, None] * C + tl.arange(0, C)[None, :], mask=mask[:, None], other=-float('inf'))
    labels = tl.load(labels_ptr + offsets, mask=mask, other=ignored_index)

    # Compute LSE
    max_logits = tl.max(logits, axis=1)
    logits_exp = tl.exp(logits - max_logits[:, None])
    sum_exp = tl.sum(logits_exp, axis=1)
    lse = max_logits + tl.log(sum_exp)

    # Compute smoothed probabilities
    smooth_prob = (1 - smoothing) * tl.where(labels[:, None] == tl.arange(0, C)[None, :], 1.0, 0.0) + smoothing / C

    # Compute cross-entropy loss
    log_probs = logits - lse[:, None]
    loss = -tl.sum(smooth_prob * log_probs, axis=1)

    # Apply LSE regularization
    loss += lse_square_scale * lse * lse

    # Ignore certain labels
    loss = tl.where(labels == ignored_index, 0.0, loss)

    # Write results back
    tl.store(loss_ptr + offsets, loss, mask=mask)
    tl.store(lse_ptr + offsets, lse, mask=mask)

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr,  # pointer to logits tensor
    labels_ptr,  # pointer to labels tensor
    lse_ptr,     # pointer to LSE tensor
    grad_output_ptr,  # pointer to gradient of output tensor
    grad_logits_ptr,  # pointer to gradient of logits tensor
    smoothing,   # label smoothing factor
    lse_square_scale,  # LSE regularization scale
    ignored_index,  # ignored label index
    N,  # number of samples
    C,  # number of classes
    BLOCK_SIZE: tl.constexpr  # block size
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    logits = tl.load(logits_ptr + offsets[:, None] * C + tl.arange(0, C)[None, :], mask=mask[:, None], other=-float('inf'))
    labels = tl.load(labels_ptr + offsets, mask=mask, other=ignored_index)
    lse = tl.load(lse_ptr + offsets, mask=mask, other=0.0)
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)

    # Compute smoothed probabilities
    smooth_prob = (1 - smoothing) * tl.where(labels[:, None] == tl.arange(0, C)[None, :], 1.0, 0.0) + smoothing / C

    # Compute gradient of logits
    grad_logits = smooth_prob - tl.exp(logits - lse[:, None])
    grad_logits *= grad_output[:, None]

    # Apply LSE regularization
    grad_logits += 2 * lse_square_scale * lse[:, None] * tl.exp(logits - lse[:, None])

    # Ignore certain labels
    grad_logits = tl.where(labels[:, None] == ignored_index, 0.0, grad_logits)

    # Write results back
    tl.store(grad_logits_ptr + offsets[:, None] * C + tl.arange(0, C)[None, :], grad_logits, mask=mask[:, None])

import torch
from torch.autograd import Function

class CrossEntropyLoss(Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing, lse_square_scale, ignored_index, process_group=None):
        N, C = logits.size()
        device = logits.device

        loss = torch.empty(N, device=device)
        lse = torch.empty(N, device=device)

        grid = (triton.cdiv(N, 1024),)
        cross_entropy_fwd_kernel[grid](
            logits, labels, loss, lse, smoothing, lse_square_scale, ignored_index, N, C, BLOCK_SIZE=1024
        )

        ctx.save_for_backward(logits, labels, lse)
        ctx.smoothing = smoothing
        ctx.lse_square_scale = lse_square_scale
        ctx.ignored_index = ignored_index
        ctx.process_group = process_group

        return loss.mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, lse = ctx.saved_tensors
        N, C = logits.size()
        device = logits.device

        grad_logits = torch.empty_like(logits)

        grid = (triton.cdiv(N, 1024),)
        cross_entropy_bwd_kernel[grid](
            logits, labels, lse, grad_output, grad_logits, ctx.smoothing, ctx.lse_square_scale, ctx.ignored_index, N, C, BLOCK_SIZE=1024
        )

        if ctx.process_group is not None:
            torch.distributed.all_reduce(grad_logits, group=ctx.process_group)

        return grad_logits, None, None, None, None, None

def cross_entropy_loss(logits, labels, smoothing=0.0, lse_square_scale=0.0, ignored_index=-100, process_group=None):
    return CrossEntropyLoss.apply(logits, labels, smoothing, lse_square_scale, ignored_index, process_group)

import torch

# Example data
logits = torch.randn(10, 5, requires_grad=True, device='cuda')
labels = torch.randint(0, 5, (10,), device='cuda')

# Compute loss
loss = cross_entropy_loss(logits, labels, smoothing=0.1, lse_square_scale=0.01, ignored_index=-100)
loss.backward()

print("Loss:", loss.item())
print("Gradients:", logits.grad)
