import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr,  # pointer to logits (N, C)
    labels_ptr,  # pointer to labels (N)
    loss_ptr,    # pointer to loss (N)
    lse_ptr,     # pointer to log-sum-exp (N)
    N,           # number of samples
    C,           # number of classes
    smoothing,   # label smoothing factor
    ignored_index,  # index to ignore
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load logits and labels
    logits = tl.load(logits_ptr + offsets[:, None] * C + tl.arange(0, C), mask=mask[:, None], other=-float('inf'))
    labels = tl.load(labels_ptr + offsets, mask=mask, other=0)

    # Compute log-sum-exp
    max_logits = tl.max(logits, axis=1)
    logits = logits - max_logits[:, None]
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=1)
    lse = max_logits + tl.log(sum_exp_logits)

    # Compute cross-entropy loss
    valid_mask = (labels != ignored_index) & mask
    valid_labels = tl.where(valid_mask, labels, 0)
    valid_logits = tl.load(logits_ptr + offsets[:, None] * C + valid_labels[:, None], mask=valid_mask[:, None], other=0)
    valid_lse = tl.load(lse_ptr + offsets, mask=valid_mask, other=0)
    loss = valid_lse - valid_logits

    # Apply label smoothing
    if smoothing > 0:
        smoothed_loss = -tl.sum(logits * (1 - smoothing) / C, axis=1)
        loss = (1 - smoothing) * loss + smoothing * smoothed_loss

    # Store results
    tl.store(loss_ptr + offsets, loss, mask=valid_mask)
    tl.store(lse_ptr + offsets, lse, mask=mask)

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr,  # pointer to logits (N, C)
    labels_ptr,  # pointer to labels (N)
    grad_output_ptr,  # pointer to gradient of loss (N)
    grad_logits_ptr,  # pointer to gradient of logits (N, C)
    N,           # number of samples
    C,           # number of classes
    smoothing,   # label smoothing factor
    ignored_index,  # index to ignore
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load logits, labels, and grad_output
    logits = tl.load(logits_ptr + offsets[:, None] * C + tl.arange(0, C), mask=mask[:, None], other=-float('inf'))
    labels = tl.load(labels_ptr + offsets, mask=mask, other=0)
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0)

    # Compute softmax
    max_logits = tl.max(logits, axis=1)
    logits = logits - max_logits[:, None]
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=1)
    softmax = exp_logits / sum_exp_logits[:, None]

    # Compute gradients
    valid_mask = (labels != ignored_index) & mask
    valid_labels = tl.where(valid_mask, labels, 0)
    valid_softmax = tl.load(softmax + offsets[:, None] * C + valid_labels[:, None], mask=valid_mask[:, None], other=0)
    grad_logits = softmax - tl.where(valid_mask[:, None], valid_softmax, 0)
    grad_logits = grad_logits * grad_output[:, None]

    # Apply label smoothing
    if smoothing > 0:
        grad_logits = grad_logits * (1 - smoothing) + smoothing / C

    # Store results
    tl.store(grad_logits_ptr + offsets[:, None] * C + tl.arange(0, C), grad_logits, mask=mask[:, None])

import torch
import triton
import triton.language as tl

class CrossEntropyLoss:
    def __init__(self, smoothing=0.0, ignored_index=-100, process_group=None):
        self.smoothing = smoothing
        self.ignored_index = ignored_index
        self.process_group = process_group

    def forward(self, logits, labels):
        N, C = logits.shape
        loss = torch.empty(N, device=logits.device)
        lse = torch.empty(N, device=logits.device)

        grid = (triton.cdiv(N, 1024),)
        cross_entropy_fwd_kernel[grid](
            logits, labels, loss, lse, N, C, self.smoothing, self.ignored_index, BLOCK_SIZE=1024
        )

        # Reduce loss across process group if distributed
        if self.process_group is not None:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM, group=self.process_group)
            loss /= torch.distributed.get_world_size(self.process_group)

        return loss, lse

    def backward(self, logits, labels, grad_output):
        N, C = logits.shape
        grad_logits = torch.empty_like(logits)

        grid = (triton.cdiv(N, 1024),)
        cross_entropy_bwd_kernel[grid](
            logits, labels, grad_output, grad_logits, N, C, self.smoothing, self.ignored_index, BLOCK_SIZE=1024
        )

        return grad_logits

def cross_entropy_loss(logits, labels, smoothing=0.0, ignored_index=-100, process_group=None):
    criterion = CrossEntropyLoss(smoothing, ignored_index, process_group)
    loss, lse = criterion.forward(logits, labels)
    return loss, lse

import torch

# Example inputs
logits = torch.randn(100, 10, device='cuda')
labels = torch.randint(0, 10, (100,), device='cuda')

# Compute forward pass
loss, lse = cross_entropy_loss(logits, labels, smoothing=0.1, ignored_index=-100)

# Compute backward pass
loss.backward()
