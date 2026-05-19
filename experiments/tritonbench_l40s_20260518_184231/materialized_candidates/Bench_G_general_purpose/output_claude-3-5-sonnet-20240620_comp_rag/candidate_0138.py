import torch
import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    # Pointers to input/output tensors
    loss_ptr, lse_ptr, z_loss_ptr, logits_ptr, labels_ptr,
    # Scalar parameters
    smoothing, logit_scale, lse_square_scale, ignored_index,
    total_classes, class_start_idx,
    # Shape information
    n_cols, n_rows, logits_row_stride,
    # Compile-time constants
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
    
    # Compute sum of logits if smoothing is enabled
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    
    # Compute log-sum-exp
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    
    # Handle ignored labels
    if label_idx == ignored_index:
        loss = 0.0
        z_loss = 0.0
    else:
        # Adjust label index for tensor parallelism
        label_idx -= class_start_idx
        
        # Check if label is in current block
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
                
        # Compute z_loss if not in split mode
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
            
    # Store results
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)
