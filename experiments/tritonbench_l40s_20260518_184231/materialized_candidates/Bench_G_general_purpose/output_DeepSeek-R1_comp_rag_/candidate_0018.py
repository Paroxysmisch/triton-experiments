import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd, custom_bwd

@triton.heuristics({'HAS_SMOOTHING': lambda args: args['smoothing'] > 0.0})
@triton.jit
def cross_entropy_forward_kernel(
    # Pointers
    logits_ptr, labels_ptr, loss_ptr, lse_ptr, 
    # Dimensions
    n_rows, n_cols, logits_row_stride,
    # Parameters
    smoothing: tl.float32, logit_scale: tl.float32,
    lse_square_scale: tl.float32, ignored_index: tl.int32,
    total_classes: tl.int32, class_start_idx: tl.int32,
    # Meta
    BLOCK_SIZE: tl.constexpr, HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_block = tl.program_id(1)
    
    # Offsets and pointers
    row_offset = row_idx * logits_row_stride
    col_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load logits and scale
    logits = tl.load(logits_ptr + row_offset + col_offsets, 
                    mask=mask, other=-float('inf')) * logit_scale
    
    # Compute LSE (Log-Sum-Exp)
    max_logit = tl.max(logits, axis=0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logit), axis=0)) + max_logit
    tl.store(lse_ptr + row_idx * n_cols + col_block, lse)
    
    # Load label and check ignored
    label = tl.load(labels_ptr + row_idx)
    if label == ignored_index:
        loss = 0.0
    else:
        # Calculate smoothed loss
        if HAS_SMOOTHING:
            smooth_pos = 1.0 - smoothing
            smooth_neg = smoothing / tl.cast(total_classes - 1, tl.float32)
        else:
            smooth_pos = 1.0
            smooth_neg = 0.0
        
        # Find target logit
        target_col = label - class_start_idx
        if target_col >= 0 and target_col < n_cols:
            target_logit = tl.load(logits_ptr + row_offset + target_col) * logit_scale
            loss = (smooth_neg * tl.sum(logits, axis=0) - smooth_pos * target_logit) 
        else:
            loss = smooth_neg * tl.sum(logits, axis=0)
        
        # Add LSE regularization if not split
        if not SPLIT:
            loss += lse * (1.0 - smooth_pos - smooth_neg * tl.cast(n_cols - 1, tl.float32))
            loss += lse_square_scale * lse * lse
    
    tl.store(loss_ptr + row_idx * n_cols + col_block, loss)

@triton.jit
def cross_entropy_backward_kernel(
    dlogits_ptr, grad_loss_ptr, logits_ptr, lse_ptr, labels_ptr,
    n_rows, n_cols, logits_row_stride,
    smoothing: tl.float32, logit_scale: tl.float32,
    lse_square_scale: tl.float32, ignored_index: tl.int32,
    total_classes: tl.int32, class_start_idx: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_block = tl.program_id(1)
    
    row_offset = row_idx * logits_row_stride
    col_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load logits and compute probs
    logits = tl.load(logits_ptr + row_offset + col_offsets, mask=mask) * logit_scale
    lse = tl.load(lse_ptr + row_idx * n_cols + col_block)
    probs = tl.exp(logits - lse)
    
    # Load label and gradient
    label = tl.load(labels_ptr + row_idx)
    grad_loss = tl.load(grad_loss_ptr + row_idx)
    
    # Calculate gradient components
    if label == ignored_index:
        grad = 0.0
    else:
        target_col = label - class_start_idx
        is_target = col_offsets == target_col
        
        # Handle label smoothing
        smooth_pos = 1.0 - smoothing if smoothing > 0 else 1.0
        smooth_neg = -smoothing / (total_classes - 1) if smoothing > 0 else 0.0
        
        grad = tl.where(is_target, 
                       (probs - smooth_pos) * grad_loss,
                       (probs + smooth_neg) * grad_loss)
        
        # Add LSE regularization gradient
        grad += 2 * lse_square_scale * lse * probs * grad_loss
    
    tl.store(dlogits_ptr + row_offset + col_offsets, 
            grad * logit_scale, mask=mask)

class FusedCrossEntropy(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, logits, labels, smoothing=0.0, logit_scale=1.0, 
                lse_reg=0.0, ignored_index=-100, process_group=None):
        # Validate inputs
        assert logits.dim() == 2, "Logits must be 2D"
        assert labels.dim() == 1, "Labels must be 1D"
        
        n_rows, n_cols = logits.shape
        logits = logits.contiguous()
        
        # Distributed setup
        world_size = 1 if process_group is None else torch.distributed.get_world_size()
        rank = 0 if process_group is None else torch.distributed.get_rank()
        split = world_size > 1
        
        # Allocate outputs
        loss = torch.empty_like(logits)
        lse = torch.empty_like(logits)
        
        # Kernel configuration
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows, triton.cdiv(n_cols, BLOCK_SIZE))
        
        # Launch forward kernel
        cross_entropy_forward_kernel[grid](
            logits, labels, loss, lse,
            n_rows, n_cols, logits.stride(0),
            smoothing, logit_scale, lse_reg, ignored_index,
            n_cols * world_size, n_cols * rank,
            BLOCK_SIZE=BLOCK_SIZE, SPLIT=split
        )
        
        # Handle distributed reduction
        if split:
            # All-gather LSE and reduce loss
            lse_all = torch.empty(world_size, *lse.shape, device=lse.device)
            torch.distributed.all_gather_into_tensor(lse_all, lse, group=process_group)
            lse = torch.logsumexp(lse_all, dim=0)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
        
        # Final loss calculation
        loss = lse - loss.mean(dim=-1)
        ctx.save_for_backward(logits, lse, labels)
        ctx.params = (smoothing, logit_scale, lse_reg, ignored_index, 
                     n_cols * world_size, n_cols * rank)
        
        return loss

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_loss):
        logits, lse, labels = ctx.saved_tensors
        smoothing, logit_scale, lse_reg, ignored_index, total_classes, class_start = ctx.params
        
        dlogits = torch.empty_like(logits)
        n_rows, n_cols = logits.shape
        
        # Kernel configuration
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows, triton.cdiv(n_cols, BLOCK_SIZE))
        
        # Launch backward kernel
        cross_entropy_backward_kernel[grid](
            dlogits, grad_loss, logits, lse, labels,
            n_rows, n_cols, logits.stride(0),
            smoothing, logit_scale, lse_reg, ignored_index,
            total_classes, class_start,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return dlogits, None, None, None, None, None, None

def fused_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_reg: float = 0.0,
    ignored_index: int = -100,
    process_group=None
) -> torch.Tensor:
    """
    Fused cross-entropy loss with Triton acceleration.
    
    Args:
        logits: (Batch, Classes) unnormalized log probabilities
        labels: (Batch,) target class indices
        label_smoothing: epsilon for label smoothing
        logit_scale: temperature scaling factor
        lse_reg: L2 regularization strength on log-sum-exp
        ignored_index: target value that ignores loss computation
        process_group: distributed process group for tensor parallelism
        
    Returns:
        loss: scalar tensor containing the computed loss
    """
    return FusedCrossEntropy.apply(
        logits, labels, label_smoothing, logit_scale, 
        lse_reg, ignored_index, process_group
    )
