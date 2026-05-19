import torch
import triton
import triton.language as tl

# Forward kernel for KL divergence computation
@triton.jit
def _kldiv_kernel_forward(
    y_pred_ptr, y_true_ptr, loss_ptr,
    batch_size, vocab_size,
    stride_b_pred, stride_v_pred,
    stride_b_true, stride_v_true,
    stride_b_loss, stride_v_loss,
    log_target, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    batch_idx = pid
    
    # Check if we're within bounds
    if batch_idx >= batch_size:
        return
        
    # Compute memory offsets
    pred_offset = batch_idx * stride_b_pred
    true_offset = batch_idx * stride_b_true
    loss_offset = batch_idx * stride_b_loss
    
    # Initialize accumulator for batch reduction
    batch_loss = 0.0
    
    # Process vocab dimension in blocks
    for v_start in range(0, vocab_size, BLOCK_SIZE):
        v_end = min(v_start + BLOCK_SIZE, vocab_size)
        n_elements = v_end - v_start
        
        # Load predictions and targets
        pred_block = tl.load(y_pred_ptr + pred_offset + v_start * stride_v_pred,
                           mask=v_start + tl.arange(0, BLOCK_SIZE) < vocab_size)
        true_block = tl.load(y_true_ptr + true_offset + v_start * stride_v_true,
                           mask=v_start + tl.arange(0, BLOCK_SIZE) < vocab_size)
        
        # Compute KL divergence based on log_target flag
        if log_target:
            # y_true is already in log space
            true_exp = tl.exp(true_block)
            loss = true_exp * (true_block - pred_block)
        else:
            # Need to compute log(y_true)
            true_safe = tl.maximum(true_block, eps)
            loss = true_block * (tl.log(true_safe) - pred_block)
            
        # Store or accumulate results
        tl.store(loss_ptr + loss_offset + v_start * stride_v_loss,
                loss, mask=v_start + tl.arange(0, BLOCK_SIZE) < vocab_size)
        batch_loss += tl.sum(loss, axis=0)

# Backward kernel for gradient computation
@triton.jit
def _kldiv_kernel_backward(
    grad_output_ptr, target_ptr, grad_input_ptr,
    batch_size, vocab_size,
    stride_b_grad, stride_v_grad,
    stride_b_target, stride_v_target,
    stride_b_out, stride_v_out,
    log_target, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid
    
    if batch_idx >= batch_size:
        return
        
    grad_offset = batch_idx * stride_b_grad
    target_offset = batch_idx * stride_b_target
    out_offset = batch_idx * stride_b_out
    
    for v_start in range(0, vocab_size, BLOCK_SIZE):
        v_end = min(v_start + BLOCK_SIZE, vocab_size)
        
        # Load gradients and targets
        grad_block = tl.load(grad_output_ptr + grad_offset + v_start * stride_v_grad,
                           mask=v_start + tl.arange(0, BLOCK_SIZE) < vocab_size)
        target_block = tl.load(target_ptr + target_offset + v_start * stride_v_target,
                             mask=v_start + tl.arange(0, BLOCK_SIZE) < vocab_size)
        
        # Compute gradients
        if log_target:
            grad = -tl.exp(target_block) * grad_block
        else:
            grad = -target_block * grad_block
            
        # Store gradients
        tl.store(grad_input_ptr + out_offset + v_start * stride_v_out,
                 grad, mask=v_start + tl.arange(0, BLOCK_SIZE) < vocab_size)

# Wrapper function for forward pass
def kldiv_forward_triton(y_pred, y_true, log_target=False, reduction='mean', eps=1e-8):
    batch_size, vocab_size = y_pred.shape
    device = y_pred.device
    
    # Initialize output tensor
    loss = torch.empty_like(y_pred)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_warps = 4
    grid = (batch_size,)
    
    _kldiv_kernel_forward[grid](
        y_pred, y_true, loss,
        batch_size, vocab_size,
        y_pred.stride(0), y_pred.stride(1),
        y_true.stride(0), y_true.stride(1),
        loss.stride(0), loss.stride(1),
        log_target, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    # Apply reduction
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'batchmean':
        return loss.sum() / batch_size
    else:
        raise ValueError(f"Unknown reduction mode: {reduction}")

# Wrapper function for backward pass
def kldiv_backward_triton(grad_output, target, log_target=False, eps=1e-8):
    batch_size, vocab_size = target.shape
    device = target.device
    
    # Initialize gradient tensor
    grad_input = torch.empty_like(target)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_warps = 4
    grid = (batch_size,)
    
    _kldiv_kernel_backward[grid](
        grad_output, target, grad_input,
        batch_size, vocab_size,
        grad_output.stride(0), grad_output.stride(1),
        target.stride(0), target.stride(1),
        grad_input.stride(0), grad_input.stride(1),
        log_target, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return grad_input
