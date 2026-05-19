import torch
import triton
import triton.language as tl

@triton.jit
def _kldiv_kernel_forward(
    y_ptr,          # [B, S], prediction ptr (in log-space)
    y_stride,       # prediction stride
    gt_ptr,         # [B, S], ground truth ptr
    gt_stride,      # ground truth stride
    loss_ptr,       # [B] or [B, S] output ptr
    loss_stride,    # output stride
    n_cols,         # number of columns
    eps,            # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
    reduction: tl.constexpr = 0,
):
    # Get program ID for batch dimension
    pid = tl.program_id(0).to(tl.int64)
    
    # Compute base pointers for this program instance
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride

    # Create offset array for block processing
    base_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize loss accumulator
    loss_sum = 0.0

    # Process the columns in blocks
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        
        # Load inputs
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)

        # Compute KL divergence based on log_target
        if not log_target:
            # KL(y_true || y) = y_true * (log(y_true) - log(y))
            loss = y_true * (tl.log(tl.maximum(y_true, eps)) - y)
        else:
            # Target is already in log space
            loss = tl.exp(y_true) * (y_true - y)

        # Handle reduction
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
    new_grads_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
):
    # Get program ID for batch dimension
    pid = tl.program_id(0).to(tl.int64)

    # Compute base pointers
    target_ptr += pid * target_stride
    new_grads_ptr += pid * new_grads_stride

    # Process columns in blocks
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        # Load target values
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

        # Compute gradients based on log_target
        if not log_target:
            res = -target  # derivative of y_true * (-log(y))
        else:
            res = -tl.exp(target)  # derivative when target is in log space

        # Store gradients
        tl.store(new_grads_ptr + offsets, res, mask=mask)

def kldiv_forward_triton(y_pred, y_true, log_target=False, reduction="none", eps=1e-8):
    BT, V = y_pred.shape
    
    # Configure block size and number of warps
    BLOCK_SIZE = min(16384, triton.next_power_of_2(V))
    num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16 if BLOCK_SIZE < 32768 else 32

    # Set grid size
    grid = (BT,)

    # Map reduction string to integer mode
    reduction_modes = {"none": 0, "sum": 1, "mean": 2, "batchmean": 3}
    reduction_mode = reduction_modes[reduction]

    # Prepare output tensor
    out_shape = (BT, V) if reduction_mode == 0 else (BT,)
    output = torch.zeros(out_shape, device=y_pred.device, dtype=torch.float32)

    # Launch kernel
    _kldiv_kernel_forward[grid](
        y_pred, y_pred.stride(0),
        y_true, y_true.stride(0),
        output, output.stride(0),
        V, eps, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        log_target=log_target,
        reduction=reduction_mode,
    )

    # Apply final reduction if needed
    if reduction_mode == 3:    # batchmean
        return output.sum() / BT
    elif reduction_mode == 1:  # sum
        return output.sum()
    elif reduction_mode == 2:  # mean
        return output.sum() / (BT * V)
    return output

def kldiv_backward_triton(target, grad_output, new_grads, log_target=False):
    BT, V = target.shape
    
    # Configure block size and number of warps
    BLOCK_SIZE = min(16384, triton.next_power_of_2(V))
    num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16 if BLOCK_SIZE < 32768 else 32

    # Launch kernel
    _kldiv_kernel_backward[BT](
        target, target.stride(0),
        new_grads, new_grads.stride(0),
        V, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        log_target=log_target,
    )

    # Apply grad_output scaling if needed
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        new_grads.mul_(grad_output)
    
    return new_grads
