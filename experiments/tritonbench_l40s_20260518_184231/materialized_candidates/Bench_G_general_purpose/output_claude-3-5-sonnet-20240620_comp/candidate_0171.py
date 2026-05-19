import triton
import triton.language as tl
import torch
import math

# Constants
MAX_FUSED_SIZE = 65536  # Maximum size for fused operation

@triton.jit
def _cross_entropy_forward_kernel(
    logits_ptr, labels_ptr, losses_ptr, lse_ptr,
    stride_b, stride_n, n_cols,
    BLOCK_SIZE: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    SCALE_LOGITS: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute batch and sequence position
    batch_idx = pid // stride_b
    seq_idx = (pid % stride_b) // stride_n
    
    # Load logits for this position
    offs_logits = tl.arange(0, BLOCK_SIZE)
    mask = offs_logits < n_cols
    
    # Base pointer for this batch,sequence position
    base_ptr = batch_idx * stride_b + seq_idx * stride_n
    logits = tl.load(logits_ptr + base_ptr + offs_logits, mask=mask, other=-float('inf'))
    
    # Apply softcap if enabled
    if USE_SOFTCAP:
        logits = tl.where(logits > 20.0, 20.0, logits)
        logits = tl.where(logits < -20.0, -20.0, logits)
    
    # Scale logits if enabled
    if SCALE_LOGITS:
        logits = logits * 0.5
    
    # Compute max for numerical stability
    max_logit = tl.max(logits, axis=0)
    
    # Compute log-sum-exp
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    log_sum_exp = tl.log(sum_exp) + max_logit
    
    # Store log-sum-exp result
    tl.store(lse_ptr + pid, log_sum_exp)
    
    # Load label
    label = tl.load(labels_ptr + pid)
    
    # Compute loss only for valid labels (not -100)
    valid_label = label != -100
    if valid_label:
        label_logit = tl.load(logits_ptr + base_ptr + label)
        loss = log_sum_exp - label_logit
        tl.store(losses_ptr + pid, loss)
    else:
        tl.store(losses_ptr + pid, 0.0)

@triton.jit
def _cross_entropy_backward_kernel(
    grad_output_ptr, logits_ptr, labels_ptr, grad_logits_ptr,
    stride_b, stride_n, n_cols,
    BLOCK_SIZE: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    SCALE_LOGITS: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Compute indices
    batch_idx = pid // stride_b
    seq_idx = (pid % stride_b) // stride_n
    
    # Load data
    offs_logits = tl.arange(0, BLOCK_SIZE)
    mask = offs_logits < n_cols
    base_ptr = batch_idx * stride_b + seq_idx * stride_n
    
    logits = tl.load(logits_ptr + base_ptr + offs_logits, mask=mask, other=-float('inf'))
    label = tl.load(labels_ptr + pid)
    grad_output = tl.load(grad_output_ptr + pid)
    
    # Apply transformations if enabled
    if USE_SOFTCAP:
        logits = tl.where(logits > 20.0, 20.0, logits)
        logits = tl.where(logits < -20.0, -20.0, logits)
    if SCALE_LOGITS:
        logits = logits * 0.5
    
    # Compute softmax
    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    softmax_output = exp_logits / sum_exp
    
    # Compute gradients
    valid_label = label != -100
    if valid_label:
        # One-hot encoding for true label
        label_mask = offs_logits == label
        grad_logits = softmax_output - label_mask.to(tl.float32)
        grad_logits = grad_logits * grad_output
    else:
        grad_logits = tl.zeros_like(softmax_output)
    
    # Store gradients
    tl.store(grad_logits_ptr + base_ptr + offs_logits, grad_logits, mask=mask)

def calculate_settings(n):
    """Calculate optimal block size and number of warps based on vocabulary size."""
    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(n))
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    elif block_size >= 4096:
        num_warps = 16
    return block_size, num_warps

class Fast_CrossEntropyLoss(torch.nn.Module):
    def __init__(self, use_softcap=False, scale_logits=False):
        super().__init__()
        self.use_softcap = use_softcap
        self.scale_logits = scale_logits
    
    def forward(self, logits, labels):
        vocab_size = logits.shape[-1]
        batch_size = logits.shape[0]
        seq_len = logits.shape[1]
        
        # Reshape inputs
        logits_2d = logits.view(-1, vocab_size)
        labels_1d = labels.view(-1)
        
        # Initialize output tensors
        losses = torch.zeros(batch_size * seq_len, device=logits.device)
        lse = torch.zeros(batch_size * seq_len, device=logits.device)
        
        # Calculate kernel settings
        block_size, num_warps = calculate_settings(vocab_size)
        
        # Launch kernel
        grid = (batch_size * seq_len,)
        _cross_entropy_forward_kernel[grid](
            logits_2d, labels_1d, losses, lse,
            logits_2d.stride(0), 1, vocab_size,
            BLOCK_SIZE=block_size,
            USE_SOFTCAP=self.use_softcap,
            SCALE_LOGITS=self.scale_logits,
            num_warps=num_warps
        )
        
        # Calculate mean loss (excluding masked positions)
        valid_mask = labels_1d != -100
        return losses[valid_mask].mean()
    
    def backward(self, grad_output):
        vocab_size = self.saved_tensors[0].shape[-1]
        batch_size = self.saved_tensors[0].shape[0]
        seq_len = self.saved_tensors[0].shape[1]
        
        # Initialize gradient tensor
        grad_logits = torch.zeros_like(self.saved_tensors[0])
        
        # Calculate kernel settings
        block_size, num_warps = calculate_settings(vocab_size)
        
        # Launch backward kernel
        grid = (batch_size * seq_len,)
        _cross_entropy_backward_kernel[grid](
            grad_output, self.saved_tensors[0].view(-1, vocab_size),
            self.saved_tensors[1].view(-1), grad_logits.view(-1, vocab_size),
            grad_logits.stride(0), 1, vocab_size,
            BLOCK_SIZE=block_size,
            USE_SOFTCAP=self.use_softcap,
            SCALE_LOGITS=self.scale_logits,
            num_warps=num_warps
        )
        
        return grad_logits, None

def fast_cross_entropy_loss(logits, labels, use_softcap=False, scale_logits=False):
    """Functional interface for fast cross entropy loss."""
    return Fast_CrossEntropyLoss(use_softcap, scale_logits)(logits, labels)
