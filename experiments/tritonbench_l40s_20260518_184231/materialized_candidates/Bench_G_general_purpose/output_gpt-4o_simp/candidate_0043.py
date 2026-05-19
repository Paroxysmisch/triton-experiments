import triton
import triton.language as tl

# Define constants for block size and number of warps
BLOCK_SIZE = 128
NUM_WARPS = 4

@triton.jit
def _kldiv_kernel_forward(y_pred_ptr, y_true_ptr, output_ptr, BT, V, log_target, eps, reduction, **meta):
    pid = tl.program_id(0)
    # Compute offsets for the block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BT * V

    # Load data
    y_pred = tl.load(y_pred_ptr + offsets, mask=mask)
    y_true = tl.load(y_true_ptr + offsets, mask=mask)

    # Compute KL divergence
    if log_target:
        kl_div = y_true - y_pred + tl.exp(y_pred - y_true)
    else:
        kl_div = y_true * (tl.log(y_true + eps) - y_pred)

    # Apply reduction
    if reduction == "none":
        tl.store(output_ptr + offsets, kl_div, mask=mask)
    elif reduction in ["sum", "mean", "batchmean"]:
        kl_div = tl.sum(kl_div, axis=0)
        if reduction == "mean":
            kl_div /= BT * V
        elif reduction == "batchmean":
            kl_div /= BT
        if pid == 0:  # Only one thread writes the result
            tl.store(output_ptr, kl_div)

def kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps):
    BT, V = y_pred.shape
    output = torch.empty_like(y_pred)
    grid = (triton.cdiv(BT * V, BLOCK_SIZE),)
    _kldiv_kernel_forward[grid](y_pred, y_true, output, BT, V, log_target, eps, reduction, num_warps=NUM_WARPS)
    return output

@triton.jit
def _kldiv_kernel_backward(y_pred_ptr, y_true_ptr, grad_output_ptr, new_grads_ptr, BT, V, log_target, **meta):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BT * V

    # Load data
    y_pred = tl.load(y_pred_ptr + offsets, mask=mask)
    y_true = tl.load(y_true_ptr + offsets, mask=mask)
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)

    # Compute gradient
    if log_target:
        grad = -tl.exp(y_pred - y_true) + 1
    else:
        grad = -y_true / (tl.exp(y_pred) + eps)

    # Apply gradient and store
    grad *= grad_output
    tl.store(new_grads_ptr + offsets, grad, mask=mask)

def kldiv_backward_triton(y_pred, y_true, grad_output, log_target):
    BT, V = y_pred.shape
    new_grads = torch.empty_like(y_pred)
    grid = (triton.cdiv(BT * V, BLOCK_SIZE),)
    _kldiv_kernel_backward[grid](y_pred, y_true, grad_output, new_grads, BT, V, log_target, num_warps=NUM_WARPS)
    return new_grads
