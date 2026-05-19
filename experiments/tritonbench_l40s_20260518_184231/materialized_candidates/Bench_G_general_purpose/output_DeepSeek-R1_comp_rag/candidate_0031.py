import torch
import triton
import triton.language as tl

@triton.jit
def _kldiv_kernel_forward(
    y_ptr,          # Pointer to predicted log-probabilities tensor [BT, V]
    y_stride,       # Stride for y_ptr
    gt_ptr,         # Pointer to ground truth tensor [BT, V]
    gt_stride,      # Stride for gt_ptr
    loss_ptr,       # Output tensor pointer
    loss_stride,    # Stride for output
    n_cols,         # Number of columns (V)
    eps,            # Numerical stability epsilon
    BLOCK_SIZE: tl.constexpr,  # Block size for processing
    log_target: tl.constexpr, # Whether target is log-form
    reduction: tl.constexpr,  # Reduction mode
):
    pid = tl.program_id(0).to(tl.int64)  # Batch index
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride

    base_offsets = tl.arange(0, BLOCK_SIZE)
    loss_sum = 0.0

    # Process columns in blocks
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)

        # Compute KL divergence
        if not log_target:
            loss = y_true * (tl.log(tl.maximum(y_true, eps)) - y)
        else:
            loss = tl.exp(y_true) * (y_true - y)

        # Store or accumulate loss based on reduction
        if reduction == 0:  # "none"
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss_sum += tl.sum(loss, axis=0)

    # Store summed loss for reduction modes
    if reduction != 0:
        tl.store(loss_ptr, loss_sum)

@triton.jit
def _kldiv_kernel_backward(
    target_ptr,         # Ground truth tensor pointer
    target_stride,      # Stride for target
    new_grads_ptr,      # Gradient output tensor pointer
    new_grads_stride,   # Stride for gradients
    n_cols,             # Number of columns (V)
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)  # Batch index
    target_ptr += pid * target_stride
    new_grads_ptr += pid * new_grads_stride

    # Process columns in blocks
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

        # Compute gradient based on log_target flag
        if not log_target:
            res = -target
        else:
            res = -tl.exp(target)

        tl.store(new_grads_ptr + offsets, res, mask=mask)

def kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps):
    BT, V = y_pred.shape
    BLOCK_SIZE = min(16384, triton.next_power_of_2(V))
    num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16 if BLOCK_SIZE < 32768 else 32
    grid = (BT,)
    reduction_code = {"none": 0, "sum": 1, "mean": 2, "batchmean": 3}[reduction]

    # Initialize output tensor
    output_shape = (BT, V) if reduction == "none" else (BT,)
    output = torch.zeros(output_shape, device=y_pred.device, dtype=torch.float32)

    _kldiv_kernel_forward[grid](
        y_pred, y_pred.stride(0), y_true, y_true.stride(0),
        output, output.stride(0), V, eps, BLOCK_SIZE,
        log_target=log_target, reduction=reduction_code,
        num_warps=num_warps
    )

    # Apply final reduction
    if reduction == "batchmean":
        return output.sum() / BT
    elif reduction == "sum":
        return output.sum()
    elif reduction == "mean":
        return output.sum() / (BT * V)
    else:  # "none"
        return output

def kldiv_backward_triton(target, grad_output, new_grads, log_target):
    BT, V = target.shape
    BLOCK_SIZE = min(16384, triton.next_power_of_2(V))
    num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16 if BLOCK_SIZE < 32768 else 32
    grid = (BT,)

    _kldiv_kernel_backward[grid](
        target, target.stride(0), new_grads, new_grads.stride(0),
        V, BLOCK_SIZE, log_target=log_target, num_warps=num_warps
    )

    # Multiply by incoming gradients if necessary
    if torch.all(grad_output == 1.0):
        return new_grads
    return new_grads * grad_output
