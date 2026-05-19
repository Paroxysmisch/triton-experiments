import torch
import triton
import triton.language as tl

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    @triton.jit
    def _cross_entropy_forward(logits, labels, loss, num_classes, softcap=None, logit_scale=None):
        row_idx = tl.program_id(0)
        col_idx = tl.arange(0, num_classes)
        
        logits_row = logits + row_idx * num_classes
        logits_row = tl.load(logits_row + col_idx)
        
        if logit_scale is not None:
            logits_row *= logit_scale
        
        max_logits = tl.max(logits_row, axis=0)
        logits_row = logits_row - max_logits
        
        exp_logits = tl.exp(logits_row)
        sum_exp_logits = tl.sum(exp_logits, axis=0)
        
        log_sum_exp = tl.log(sum_exp_logits)
        normalized_logits = logits_row - log_sum_exp
        
        if softcap is not None:
            normalized_logits = tl.minimum(normalized_logits, softcap)
        
        label = tl.load(labels + row_idx)
        loss_val = -normalized_logits[label]
        
        tl.store(loss + row_idx, loss_val)

    @staticmethod
    @triton.jit
    def _cross_entropy_backward(logits, labels, grad_output, grad_logits, num_classes, softcap=None, logit_scale=None):
        row_idx = tl.program_id(0)
        col_idx = tl.arange(0, num_classes)
        
        logits_row = logits + row_idx * num_classes
        logits_row = tl.load(logits_row + col_idx)
        
        if logit_scale is not None:
            logits_row *= logit_scale
        
        max_logits = tl.max(logits_row, axis=0)
        logits_row = logits_row - max_logits
        
        exp_logits = tl.exp(logits_row)
        sum_exp_logits = tl.sum(exp_logits, axis=0)
        
        prob = exp_logits / sum_exp_logits
        
        label = tl.load(labels + row_idx)
        grad = prob
        grad[label] -= 1.0
        
        if softcap is not None:
            grad = tl.minimum(grad, softcap)
        
        grad *= grad_output[row_idx]
        
        tl.store(grad_logits + row_idx * num_classes + col_idx, grad)

    @staticmethod
    def forward(ctx, logits, labels, softcap=None, logit_scale=None):
        num_classes = logits.shape[1]
        loss = torch.empty(logits.shape[0], device=logits.device, dtype=logits.dtype)
        
        Fast_CrossEntropyLoss._cross_entropy_forward[(logits.shape[0],)](logits, labels, loss, num_classes, softcap, logit_scale)
        
        ctx.save_for_backward(logits, labels)
        ctx.num_classes = num_classes
        ctx.softcap = softcap
        ctx.logit_scale = logit_scale
        
        return loss.mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels = ctx.saved_tensors
        num_classes = ctx.num_classes
        softcap = ctx.softcap
        logit_scale = ctx.logit_scale
        
        grad_logits = torch.empty_like(logits)
        
        Fast_CrossEntropyLoss._cross_entropy_backward[(logits.shape[0],)](logits, labels, grad_output, grad_logits, num_classes, softcap, logit_scale)
        
        return grad_logits, None, None, None

def fast_cross_entropy_loss(logits, labels, softcap=None, logit_scale=None):
    return Fast_CrossEntropyLoss.apply(logits, labels, softcap, logit_scale)

# Example usage:
logits = torch.randn(32, 10, device='cuda', requires_grad=True)
labels = torch.randint(0, 10, (32,), device='cuda')
loss = fast_cross_entropy_loss(logits, labels)
loss.backward()
