import triton
import triton.language as tl
import torch

@triton.jit
def cross_entropy_fwd_kernel(
    # Pointers to tensors
    logits_ptr, labels_ptr,
    loss_ptr, lse_ptr, z_loss_ptr,
    # Tensor dimensions and strides
    logits_row_stride, labels_stride,
    loss_stride, lse_stride, z_loss_stride,
    n_rows, total_classes,
    # Parameters
    logit_scale: tl.constexpr,
    smoothing: tl.constexpr,
    lse_square_scale: tl.constexpr,
    ignored_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr,
    class_start_idx: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block = tl.program_id(1)
    
    # Offset calculations
    logits_row = logits_ptr + row_idx * logits_row_stride
    labels_row = labels_ptr + row_idx * labels_stride
    label = tl.load(labels_row).to(tl.int32)
    
    # Initialize pointers for outputs
    loss_row = loss_ptr + row_idx * loss_stride
    lse_row = lse_ptr + row_idx * lse_stride
    z_loss_row = z_loss_ptr + row_idx * z_loss_stride
    
    # Column offsets for this block
    col_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < total_classes
    cols = class_start_idx + col_offsets  # For partitioned vocab
    
    # Load logits and apply scaling
    logits = tl.load(logits_row + cols, mask=mask, other=-float('inf'))
    scaled_logits = logits * logit_scale
    
    # Compute LSE for this block
    max_logit = tl.max(scaled_logits, axis=0)
    lse_block = max_logit + tl.log(tl.sum(tl.exp(scaled_logits - max_logit), axis=0))
    
    # Handle label presence in this block
    label_in_block = (label >= cols[0]) & (label < cols[0] + BLOCK_SIZE)
    if label_in_block:
        label_offset = label - cols[0]
        label_logit = tl.load(logits_row + label, mask=label != ignored_index)
        label_logit_scaled = label_logit * logit_scale
    else:
        label_logit_scaled = 0.0
    
    # Calculate loss components
    loss_block = tl.where(
        label != ignored_index,
        lse_block - label_logit_scaled,
        0.0
    )
    
    # Compute z_loss component
    z_loss_block = lse_square_scale * (lse_block * lse_block)
    
    # Store block results
    tl.store(lse_row + col_block, lse_block)
    tl.store(loss_row + col_block, loss_block)
    tl.store(z_loss_row + col_block, z_loss_block)
    
    # Summing for label smoothing (if needed)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(scaled_logits, axis=0)
        tl.debug_barrier()
        # Store sum_logits in global memory if required

@triton.jit
def cross_entropy_bwd_kernel(
    # Pointers to tensors
    dlogits_ptr, logits_ptr,
    dloss_ptr, lse_ptr, labels_ptr,
    # Tensor dimensions and strides
    dlogits_row_stride, logits_row_stride,
    dloss_stride, lse_stride, labels_stride,
    n_rows, total_classes,
    # Parameters
    logit_scale: tl.constexpr,
    smoothing: tl.constexpr,
    lse_square_scale: tl.constexpr,
    ignored_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    total_classes_bwd: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block = tl.program_id(1)
    
    # Load label and check ignore
    label = tl.load(labels_ptr + row_idx * labels_stride).to(tl.int32)
    if label == ignored_index:
        return
    
    # Offset calculations
    logits_row = logits_ptr + row_idx * logits_row_stride
    dlogits_row = dlogits_ptr + row_idx * dlogits_row_stride
    lse = tl.load(lse_ptr + row_idx * lse_stride)
    dloss = tl.load(dloss_ptr + row_idx * dloss_stride)
    
    # Column offsets
    col_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < total_classes
    
    # Load logits and compute probabilities
    logits = tl.load(logits_row + col_offsets, mask=mask, other=0.0)
    scaled_logits = logits * logit_scale
    probs = tl.exp(scaled_logits - lse)
    
    # Base gradient calculation
    grad = probs
    if HAS_SMOOTHING:
        smoothing_factor = smoothing / (total_classes_bwd - 1)
        target = tl.where(col_offsets == label, 1.0 - smoothing, smoothing_factor)
    else:
        target = tl.where(col_offsets == label, 1.0, 0.0)
    grad -= target
    
    # Apply loss gradient and z_loss contribution
    total_grad = grad * dloss
    if lse_square_scale != 0:
        z_grad = 2 * lse_square_scale * lse * probs
        total_grad += z_grad * dloss
    
    # Store gradients
    tl.store(dlogits_row + col_offsets, total_grad, mask=mask)

def cross_entropy_fwd(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logit_scale: float = 1.0,
    smoothing: float = 0.0,
    lse_square_scale: float = 0.0,
    ignored_index: int = -100,
    BLOCK_SIZE: int = 1024,
):
    n_rows, total_classes = logits.shape
    grid = (n_rows, (total_classes + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    # Allocate output tensors
    loss = torch.empty((n_rows, grid[1]), device=logits.device)
    lse = torch.empty_like(loss)
    z_loss = torch.empty_like(loss)
    
    cross_entropy_fwd_kernel[grid](
        logits, labels, loss, lse, z_loss,
        logits.stride(0), labels.stride(0),
        loss.stride(0), lse.stride(0), z_loss.stride(0),
        n_rows, total_classes,
        logit_scale, smoothing, lse_square_scale, ignored_index,
        BLOCK_SIZE, smoothing > 0, False, 0
    )
    
    # Aggregate results
    final_lse = torch.logsumexp(lse, dim=1)
    final_loss = torch.sum(loss, dim=1) + final_lse
    final_z_loss = torch.sum(z_loss, dim=1)
    
    return final_loss, final_lse, final_z_loss

def cross_entropy_bwd(
    dloss: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    logit_scale: float = 1.0,
    smoothing: float = 0.0,
    lse_square_scale: float = 0.0,
    ignored_index: int = -100,
    BLOCK_SIZE: int = 1024,
):
    n_rows, total_classes = logits.shape
    grid = (n_rows, (total_classes + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    dlogits = torch.empty_like(logits)
    
    cross_entropy_bwd_kernel[grid](
        dlogits, logits, dloss, lse, labels,
        dlogits.stride(0), logits.stride(0),
        dloss.stride(0), lse.stride(0), labels.stride(0),
        n_rows, total_classes,
        logit_scale, smoothing, lse_square_scale, ignored_index,
        BLOCK_SIZE, smoothing > 0, total_classes
    )
    
    return dlogits

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_scale, smoothing, lse_square_scale, ignored_index):
        loss, lse, z_loss = cross_entropy_fwd(
            logits, labels, logit_scale, smoothing, lse_square_scale, ignored_index
        )
        ctx.save_for_backward(logits, lse, labels)
        ctx.params = (logit_scale, smoothing, lse_square_scale, ignored_index)
        return loss + z_loss  # Or handle separately

    @staticmethod
    def backward(ctx, grad_output):
        logits, lse, labels = ctx.saved_tensors
        logit_scale, smoothing, lse_square_scale, ignored_index = ctx.params
        dlogits = cross_entropy_bwd(
            grad_output, logits, labels, lse,
            logit_scale, smoothing, lse_square_scale, ignored_index
        )
        return dlogits, None, None, None, None, None

def cross_entropy_loss(logits, labels, logit_scale=1.0, smoothing=0.0, z_weight=0.0, ignored_index=-100):
    return CrossEntropyLoss.apply(logits, labels, logit_scale, smoothing, z_weight, ignored_index)
