import triton
import torch
from torch.autograd import Function

class CrossEntropyLoss(Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, lse_square_scale=0.0, ignored_index=-1, process_group=None):
        # Setup
        ctx.smoothing = smoothing
        ctx.lse_square_scale = lse_square_scale
        ctx.ignored_index = ignored_index
        ctx.process_group = process_group

        # Allocate output tensor
        losses = torch.empty_like(labels)

        # Launch kernel
        triton.kernel()(logits=logits, labels=labels, losses=losses, smoothing=smoothing, lse_square_scale=lse_square_scale, ignored_index=ignored_index, process_group=process_group)

        # Save tensors for backward pass
        ctx.save_for_backward(logits, labels)

        return losses

    @staticmethod
    def backward(ctx, grad_output):
        # Load tensors
        logits, labels = ctx.saved_tensors

        # Allocate output tensor
        grad_input = torch.empty_like(logits)

        # Launch backward kernel
        triton.kernel()(logits=logits, labels=labels, grad_input=grad_input, grad_output=grad_output, smoothing=ctx.smoothing, lse_square_scale=ctx.lse_square_scale, ignored_index=ctx.ignored_index, process_group=ctx.process_group)

        return grad_input, None, None, None, None, None

def cross_entropy_loss(logits, labels, smoothing=0.0, lse_square_scale=0.0, ignored_index=-1, process_group=None):
    return CrossEntropyLoss.apply(logits, labels, smoothing, lse_square_scale, ignored_index, process_group)
