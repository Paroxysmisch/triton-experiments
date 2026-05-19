import triton
import triton.language as tl
import torch

@triton.jit
def _cross_entropy_forward(
    logits_ptr, labels_ptr, loss_ptr, num_classes, softcap, logit_scale, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_size = logits_ptr.shape[0]
    row_start = pid * BLOCK_SIZE
    row_end = row_start + BLOCK_SIZE

    for i in range(row_start, row_end):
        if i < batch_size:
            logits = tl.load(logits_ptr + i * num_classes, mask=tl.arange(0, num_classes) < num_classes)
            if logit_scale > 1.0:
                logits = logits / logit_scale
            if softcap > 0.0:
                logits = tl.log(1 + tl.exp(logits)) / softcap
            max_logit = tl.max(logits, axis=0)
            logits = logits - max_logit
            exp_logits = tl.exp(logits)
            sum_exp_logits = tl.sum(exp_logits, axis=0)
            log_sum_exp = tl.log(sum_exp_logits)
            label = tl.load(labels_ptr + i)
            loss = log_sum_exp - logits[label]
            tl.store(loss_ptr + i, loss)

@triton.jit
def _cross_entropy_backward(
    grad_output_ptr, logits_ptr, labels_ptr, grad_logits_ptr, num_classes, softcap, logit_scale, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    batch_size = logits_ptr.shape[0]
    row_start = pid * BLOCK_SIZE
    row_end = row_start + BLOCK_SIZE

    for i in range(row_start, row_end):
        if i < batch_size:
            logits = tl.load(logits_ptr + i * num_classes, mask=tl.arange(0, num_classes) < num_classes)
            if logit_scale > 1.0:
                logits = logits / logit_scale
            if softcap > 0.0:
                logits = tl.log(1 + tl.exp(logits)) / softcap
            max_logit = tl.max(logits, axis=0)
            logits = logits - max_logit
            exp_logits = tl.exp(logits)
            sum_exp_logits = tl.sum(exp_logits, axis=0)
            label = tl.load(labels_ptr + i)
            grad_output = tl.load(grad_output_ptr + i)
            grad = exp_logits / sum_exp_logits
            grad[label] -= 1.0
            grad *= grad_output
            if logit_scale > 1.0:
                grad *= logit_scale
            if softcap > 0.0:
                grad *= (1 - tl.exp(logits) / (1 + tl.exp(logits))) / softcap
            tl.store(grad_logits_ptr + i * num_classes, grad, mask=tl.arange(0, num_classes) < num_classes)

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, softcap=0.0, logit_scale=1.0):
        batch_size, num_classes = logits.shape
        loss = torch.empty(batch_size, device=logits.device, dtype=logits.dtype)
        
        # Launch the forward kernel
        grid = (batch_size, )
        _cross_entropy_forward[grid](logits, labels, loss, num_classes, softcap, logit_scale, BLOCK_SIZE=1024)
        
        ctx.save_for_backward(logits, labels)
        ctx.softcap = softcap
        ctx.logit_scale = logit_scale
        return loss.mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels = ctx.saved_tensors
        batch_size, num_classes = logits.shape
        grad_logits = torch.zeros_like(logits)
        
        # Launch the backward kernel
        grid = (batch_size, )
        _cross_entropy_backward[grid](grad_output, logits, labels, grad_logits, num_classes, ctx.softcap, ctx.logit_scale, BLOCK_SIZE=1024)
        
        return grad_logits, None, None, None

def fast_cross_entropy_loss(logits, labels, softcap=0.0, logit_scale=1.0):
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    valid_mask = labels >= 0
    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask]
    loss = Fast_CrossEntropyLoss.apply(valid_logits, valid_labels, softcap, logit_scale)
    return loss

import torch

# Example inputs
logits = torch.randn(10, 5, requires_grad=True, device='cuda')
labels = torch.randint(0, 5, (10,), device='cuda')

# Compute the loss
loss = fast_cross_entropy_loss(logits, labels, softcap=0.1, logit_scale=2.0)
print(loss)

# Backward pass
loss.backward()
print(logits.grad)
