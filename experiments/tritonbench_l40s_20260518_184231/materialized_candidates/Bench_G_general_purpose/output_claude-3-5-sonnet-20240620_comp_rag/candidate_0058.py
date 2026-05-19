import torch
import triton
import triton.language as tl

@triton.jit
def _kldiv_kernel_forward(
    y_ptr,          # prediction pointer (log-space)
    y_stride,       # prediction stride
    gt_ptr,         # ground truth pointer
    gt_stride,      # ground truth stride 
    loss_ptr,       # output pointer
    loss_stride,    # output stride
    n_cols,         # number of columns
    eps,            # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
    reduction: tl.constexpr = 0,
):
    # Get program ID for batch dimension
    pid = tl.program_id(0)
    
    # Offset pointers for current batch
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride

    # Create offsets for block processing
    base_offsets = tl.arange(0, BLOCK_SIZE)
    
    loss_sum = 0.0
    
    # Process columns in blocks
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        
        # Load values
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)

        # Compute KL divergence based on log_target
        if not log_target:
            # KL(y_true || y) = y_true * (log(y_true) - log(y))
            loss = y_true * (tl.log(tl.maximum(y_true, eps)) - y)
        else:
            # If target is in log-space
            loss = tl.exp(y_true) * (y_true - y)

        # Handle reduction modes
        if reduction == 0:  # none
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss_sum += tl.sum(loss, axis=0)

    if reduction != 0:
        tl.store(loss_ptr, loss_sum)

@triton.jit
def _kldiv_kernel_backward(
    target_ptr,
    target_stride,
    new_grads_ptr,
