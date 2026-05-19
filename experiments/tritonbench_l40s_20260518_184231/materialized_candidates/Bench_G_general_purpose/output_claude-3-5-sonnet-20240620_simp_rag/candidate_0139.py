import torch
import triton
import triton.language as tl
from typing import Optional, Tuple
from torch.cuda.amp import custom_fwd, custom_bwd

@triton.jit
def cross_entropy_fwd_kernel(
    loss_ptr,
    lse_ptr,
    z_loss_ptr,
    logits_ptr,
    labels_ptr,
    smoothing,  # label smoothing factor
    logit_scale,  # scaling factor for logits
    lse_square_scale,  # scaling factor for z-loss
    ignored_index,
    total_classes,
    class_start_idx,
    n_cols,
    n_rows,
    logits_row_stride,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # Get program ID for the current thread
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    
    # Calculate pointer offsets
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load label for current row
    label_idx = tl.load(labels_ptr + row_idx)
    
    # Load and scale logits
    logits = tl.load(
        logits_ptr + col_offsets, 
        mask=col_offsets < n_cols, 
        other=-float("inf")
    ).to(tl.float32) * logit_scale
    
    # Compute max for numerical stability
    max_logits = tl.max(logits, 0)
    
    # Compute sum of logits if using label smoothing
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    
    # Compute log-sum-exp
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    
    # Handle ignored indices
    if label_idx == ignored_index:
        loss = 0.0
        z_loss = 0.0
    else:
        # Adjust label index for tensor parallelism
        label_idx -= class_start_idx
        
        # Compute loss based on whether label is in current block
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(n_cols, (col_block_idx + 1) * BLOCK_SIZE):
            logits_label = tl.load(logits_ptr + label_idx) * logit_scale
            if HAS_SMOOTHING:
                loss = (
                    (lse if not SPLIT else 0.0)
                    - smoothing * sum_logits / total_classes
                    - (1 - smoothing) * logits_label
                )
            else:
                loss = (lse if not SPLIT else 0.0) - logits_label
        else:
            if HAS_SMOOTHING:
                loss = smoothing * ((lse if not SPLIT else 0.0) - sum_logits / total_classes)
            else:
                loss = 0.0
                
        # Add z-loss if not in split mode
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
            
    # Store results
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)

@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr,
    dloss_ptr,
    logits_ptr,
    lse_ptr,
    labels_ptr,
    smoothing,
    logit_scale,
    lse_square_scale,
    ignored_index,
    total_classes,
    class_start_idx,
    n_cols,
    logits_row_stride,
    dlogits_row_stride,
    dloss_row_stride,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
):
    # Get program ID for the current thread
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    
    # Calculate pointer offsets
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load label and loss gradient
    label_idx = tl.load(labels_ptr + row_idx)
    dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride) if label_idx != ignored_index else 0.0
    
    # Load and scale logits
    logits = tl.load(
        logits_ptr + col_offsets, 
        mask=col_offsets < n_cols, 
        other=-float("inf")
    ).to(tl.float32) * logit_scale
    
    # Load LSE and compute probabilities
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    
    # Add z-loss contribution
    probs += 2.0 * lse_square_scale * lse * probs
    
    # Adjust label index for tensor parallelism
    label_idx -= class_start_idx
    
    # Apply label smoothing if enabled
    if HAS_SMOOTHING:
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 - smoothing), probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    
    # Store gradients
    tl.store(dlogits_ptr + col_offsets, (dloss * logit_scale) * probs, mask=col_offsets < n_cols)

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        smoothing: float = 0.0,
        logit_scale: float = 1.0,
        z_loss_scale: float = 0.0,
        ignored_index: int = -100,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Implementation details for forward pass
        BLOCK_SIZE = 128
        n_rows, n_cols = logits.shape
        
        # Initialize output tensors
        loss = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
        lse = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
        z_loss = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
        
        # Launch kernel
        grid = (n_rows, (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE)
        cross_entropy_fwd_kernel[grid](
            loss.data_ptr(),
            lse.data_ptr(),
            z_loss.data_ptr(),
            logits.data_ptr(),
            labels.data_ptr(),
            smoothing,
            logit_scale,
            z_loss_scale,
            ignored_index,
            logits.shape[1],
            0,  # class_start_idx
            n_cols,
            n_rows,
            logits.stride(0),
            BLOCK_SIZE,
            smoothing > 0.0,
            False,  # SPLIT
        )
        
        # Save for backward
        ctx.save_for_backward(logits, labels, lse)
        ctx.constants = (smoothing, logit_scale, z_loss_scale, ignored_index)
        
        return loss.mean(), z_loss.mean()

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor, grad_z_loss: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None, None, None]:
        logits, labels, lse = ctx.saved_tensors
        smoothing, logit_scale, z_loss_scale, ignored_index = ctx.constants
        
        # Initialize gradient tensor
        grad_logits = torch.empty_like(logits)
        
        # Launch backward kernel
        BLOCK_SIZE = 128
        n_rows, n_cols = logits.shape
        grid = (n_rows, (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE)
        
        cross_entropy_bwd_kernel[grid](
            grad_logits.data_ptr(),
            grad_output.data_ptr(),
            logits.data_ptr(),
            lse.data_ptr(),
            labels.data_ptr(),
            smoothing,
            logit_scale,
            z_loss_scale,
            ignored_index,
            logits.shape[1],
            0,  # class_start_idx
            n_cols,
            logits.stride(0),
            grad_logits.stride(0),
            1,  # dloss_row_stride
            BLOCK_SIZE,
            smoothing > 0.0,
        )
        
        return grad_logits, None, None, None, None, None, None

def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.0,
    logit_scale: float = 1.0,
    z_loss_scale: float = 0.0,
    ignored_index: int = -100,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cross entropy loss with optional label smoothing and logit scaling.
    
    Args:
        logits: Input logits of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)
        smoothing: Label smoothing factor (default: 0.0)
        logit_scale: Scaling factor for logits (default: 1.0)
        z_loss_scale: Scale factor for z-loss regularization (default: 0.0)
        ignored_index: Index to ignore in loss computation (default: -100)
        process_group: Optional process group for distributed training
        
    Returns:
        Tuple of (loss, z_loss)
    """
    return CrossEntropyLoss.apply(
        logits, labels, smoothing, logit_scale, z_loss_scale, ignored_index, process_group
    )
