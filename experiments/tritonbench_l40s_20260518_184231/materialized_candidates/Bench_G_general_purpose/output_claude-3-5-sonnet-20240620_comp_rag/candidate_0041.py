import triton
import triton.language as tl
import torch

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr, lse_ptr, z_loss_ptr,
    stride, total_classes, ignored_index,
    smoothing, logit_scale, lse_square_scale,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # Get program ID for the current instance
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    # Calculate offsets
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < total_classes
    
    # Load data
    logits_offset = row_idx * stride + col_offsets
    logits = tl.load(logits_ptr + logits_offset, mask=mask, other=-float('inf'))
    label = tl.load(labels_ptr + row_idx)
    
    # Apply logit scaling
    if logit_scale != 1.0:
        logits = logits * logit_scale
    
    # Calculate log-sum-exp
    max_logit = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    lse = max_logit + tl.log(sum_exp)
    
    # Store LSE
    tl.store(lse_ptr + row_idx, lse)
    
    # Calculate loss
    if label != ignored_index:
        target_logit = tl.load(logits_ptr + row_idx * stride + label)
        if logit_scale != 1.0:
            target_logit = target_logit * logit_scale
            
        loss = lse - target_logit
        
        if HAS_SMOOTHING:
            loss = (1.0 - smoothing) * loss - smoothing * tl.log(sum_exp / total_classes)
            
        # Calculate z_loss if needed
        if lse_square_scale > 0.0:
            z_loss = lse_square_scale * (lse * lse)
            tl.store(z_loss_ptr + row_idx, z_loss)
            loss = loss + z_loss
    else:
        loss = 0.0
        if lse_square_scale > 0.0:
            tl.store(z_loss_ptr + row_idx, 0.0)
    
    tl.store(loss_ptr + row_idx, loss)

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, labels_ptr, dlogits_ptr, lse_ptr,
    stride, total_classes, ignored_index,
    smoothing, logit_scale,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr
):
    # Get indices
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    # Calculate offsets
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < total_classes
    
    # Load data
    logits_offset = row_idx * stride + col_offsets
    logits = tl.load(logits_ptr + logits_offset, mask=mask, other=-float('inf'))
    label = tl.load(labels_ptr + row_idx)
    lse = tl.load(lse_ptr + row_idx)
    
    if label != ignored_index:
        # Calculate probabilities
        if logit_scale != 1.0:
            logits = logits * logit_scale
        
        probs = tl.exp(logits - lse)
        
        # Calculate gradients
        if HAS_SMOOTHING:
            smooth_factor = smoothing / total_classes
            dlogits = probs * (1.0 - smoothing) - smooth_factor
            
            # Adjust gradient for target class
            target_mask = col_offsets == label
            dlogits = tl.where(target_mask, dlogits - (1.0 - smoothing), dlogits)
        else:
            dlogits = probs
            target_mask = col_offsets == label
            dlogits = tl.where(target_mask, dlogits - 1.0, dlogits)
        
        if logit_scale != 1.0:
            dlogits = dlogits * logit_scale
            
        tl.store(dlogits_ptr + logits_offset, dlogits, mask=mask)
    else:
        tl.store(dlogits_ptr + logits_offset, 0.0, mask=mask)

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, logit_scale=1.0, 
               lse_square_scale=0.0, ignored_index=-100):
        batch_size, vocab_size = logits.shape
        BLOCK_SIZE = min(vocab_size, 1024)
        num_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Allocate output tensors
        loss = torch.empty(batch_size, device=logits.device, dtype=torch.float32)
        lse = torch.empty_like(loss)
        z_loss = torch.empty_like(loss) if lse_square_scale > 0.0 else None
        
        # Launch kernel
        grid = (batch_size, num_blocks)
        cross_entropy_fwd_kernel[grid](
            logits, labels, loss, lse, z_loss,
            logits.stride(0), vocab_size, ignored_index,
            smoothing, logit_scale, lse_square_scale,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_SMOOTHING=smoothing > 0.0,
            SPLIT=num_blocks > 1
        )
        
        # Save for backward
        ctx.save_for_backward(logits, labels, lse)
        ctx.other_args = (smoothing, logit_scale, ignored_index, vocab_size)
        
        return loss, lse, z_loss if z_loss is not None else None
    
    @staticmethod
    def backward(ctx, grad_loss, grad_lse, grad_z_loss):
        logits, labels, lse = ctx.saved_tensors
        smoothing, logit_scale, ignored_index, vocab_size = ctx.other_args
        
        batch_size = logits.size(0)
        BLOCK_SIZE = min(vocab_size, 1024)
        num_blocks = (vocab_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Allocate gradient tensor
        dlogits = torch.empty_like(logits)
        
        # Launch kernel
        grid = (batch_size, num_blocks)
        cross_entropy_bwd_kernel[grid](
            logits, labels, dlogits, lse,
            logits.stride(0), vocab_size, ignored_index,
            smoothing, logit_scale,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_SMOOTHING=smoothing > 0.0
        )
        
        return dlogits * grad_loss.view(-1, 1), None, None, None, None, None
