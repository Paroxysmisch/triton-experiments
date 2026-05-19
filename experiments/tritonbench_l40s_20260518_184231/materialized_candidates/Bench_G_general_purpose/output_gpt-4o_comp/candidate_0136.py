import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr, lse_ptr,
    num_classes, smoothing, scale, ignore_index,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ids
    pid = tl.program_id(axis=0)
    
    # Create pointers for the current row
    row_logits_ptr = logits_ptr + pid * num_classes
    row_labels_ptr = labels_ptr + pid
    row_loss_ptr = loss_ptr + pid
    row_lse_ptr = lse_ptr + pid
    
    # Load label
    label = tl.load(row_labels_ptr)
    
    # Check if label should be ignored
    mask_ignore = label == ignore_index
    
    # Compute max logit for numerical stability
    max_logit = tl.max(tl.load(row_logits_ptr + tl.arange(0, num_classes)))
    
    # Compute log-sum-exp
    exp_logits = tl.exp(tl.load(row_logits_ptr + tl.arange(0, num_classes)) - max_logit)
    sum_exp_logits = tl.sum(exp_logits)
    lse = max_logit + tl.log(sum_exp_logits)
    
    # Store log-sum-exp
    tl.store(row_lse_ptr, lse)
    
    # Compute loss
    if not mask_ignore:
        # Apply label smoothing
        smoothed_label = (1 - smoothing) + smoothing / num_classes
        true_logit = tl.load(row_logits_ptr + label)
        loss = lse - true_logit * (1 - smoothing) - smoothed_label * scale
    else:
        loss = 0.0
    
    # Store loss
    tl.store(row_loss_ptr, loss)

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, labels_ptr, grad_output_ptr, grad_logits_ptr,
    num_classes, smoothing, scale, ignore_index,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ids
    pid = tl.program_id(axis=0)
    
    # Create pointers for the current row
    row_logits_ptr = logits_ptr + pid * num_classes
    row_labels_ptr = labels_ptr + pid
    row_grad_output_ptr = grad_output_ptr + pid
    row_grad_logits_ptr = grad_logits_ptr + pid * num_classes
    
    # Load label and gradient output
    label = tl.load(row_labels_ptr)
    grad_output = tl.load(row_grad_output_ptr)
    
    # Check if label should be ignored
    mask_ignore = label == ignore_index
    
    # Compute max logit for numerical stability
    max_logit = tl.max(tl.load(row_logits_ptr + tl.arange(0, num_classes)))
    
    # Compute log-sum-exp
    exp_logits = tl.exp(tl.load(row_logits_ptr + tl.arange(0, num_classes)) - max_logit)
    sum_exp_logits = tl.sum(exp_logits)
    probs = exp_logits / sum_exp_logits
    
    # Compute gradient
    if not mask_ignore:
        # Apply label smoothing
        smoothed_label = (1 - smoothing) + smoothing / num_classes
        true_grad = -smoothed_label * scale
        probs = probs - (tl.arange(0, num_classes) == label) * (1 - smoothing)
        grad_logits = grad_output * (probs + true_grad)
    else:
        grad_logits = 0.0
    
    # Store gradient
    tl.store(row_grad_logits_ptr + tl.arange(0, num_classes), grad_logits)

class CrossEntropyLoss:
    def __init__(self, num_classes, smoothing=0.0, scale=1.0, ignore_index=-100):
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.scale = scale
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        # Allocate memory for loss and lse
        loss = torch.empty_like(labels, dtype=logits.dtype)
        lse = torch.empty_like(labels, dtype=logits.dtype)
        
        # Launch forward kernel
        grid = (logits.shape[0],)
        cross_entropy_fwd_kernel[grid](
            logits, labels, loss, lse,
            self.num_classes, self.smoothing, self.scale, self.ignore_index,
            BLOCK_SIZE=self.num_classes
        )
        
        return loss, lse

    def backward(self, logits, labels, grad_output):
        # Allocate memory for gradient w.r.t. logits
        grad_logits = torch.empty_like(logits)
        
        # Launch backward kernel
        grid = (logits.shape[0],)
        cross_entropy_bwd_kernel[grid](
            logits, labels, grad_output, grad_logits,
            self.num_classes, self.smoothing, self.scale, self.ignore_index,
            BLOCK_SIZE=self.num_classes
        )
        
        return grad_logits

def cross_entropy_loss(logits, labels, num_classes, smoothing=0.0, scale=1.0, ignore_index=-100):
    loss_fn = CrossEntropyLoss(num_classes, smoothing, scale, ignore_index)
    loss, lse = loss_fn.forward(logits, labels)
    return loss, lse
