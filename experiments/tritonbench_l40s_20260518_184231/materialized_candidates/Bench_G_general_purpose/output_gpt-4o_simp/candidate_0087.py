import triton
import triton.language as tl

# Reduction modes
_REDUCTION_MODE_NONE = 0
_REDUCTION_MODE_SUM = 1
_REDUCTION_MODE_MEAN = 2
_REDUCTION_MODE_BATCHMEAN = 3

def get_num_warps(block_size):
    # Simple heuristic to choose the number of warps based on block size
    if block_size <= 64:
        return 1
    elif block_size <= 128:
        return 2
    elif block_size <= 256:
        return 4
    else:
        return 8

@triton.jit
def _kldiv_kernel_forward(y_pred_ptr, y_true_ptr, output_ptr, n_elements, log_target, reduction_mode, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load predictions and targets
    y_pred = tl.load(y_pred_ptr + offsets, mask=offsets < n_elements, other=0.0)
    y_true = tl.load(y_true_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Compute KL divergence
    if log_target:
        kl_div = y_true * (tl.log(y_true) - y_pred)
    else:
        kl_div = y_true * (tl.log(y_true) - tl.log(y_pred))

    # Store the result
    tl.store(output_ptr + offsets, kl_div, mask=offsets < n_elements)

def kldiv_forward_triton(y_pred, y_true, log_target=False, reduction='none', BLOCK_SIZE=256):
    # Convert reduction string to mode
    reduction_mode = {
        'none': _REDUCTION_MODE_NONE,
        'sum': _REDUCTION_MODE_SUM,
        'mean': _REDUCTION_MODE_MEAN,
        'batchmean': _REDUCTION_MODE_BATCHMEAN
    }[reduction]

    # Determine the number of elements and configure execution
    n_elements = y_pred.numel()
    num_warps = get_num_warps(BLOCK_SIZE)

    # Allocate output
    output = torch.empty_like(y_pred)

    # Launch the kernel
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    _kldiv_kernel_forward[grid](y_pred, y_true, output, n_elements, log_target, reduction_mode, BLOCK_SIZE=BLOCK_SIZE)

    # Apply reduction
    if reduction_mode == _REDUCTION_MODE_SUM:
        return output.sum()
    elif reduction_mode == _REDUCTION_MODE_MEAN:
        return output.mean()
    elif reduction_mode == _REDUCTION_MODE_BATCHMEAN:
        return output.sum() / y_pred.size(0)
    else:
        return output

@triton.jit
def _kldiv_kernel_backward(y_pred_ptr, y_true_ptr, grad_output_ptr, grad_input_ptr, n_elements, log_target, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load predictions, targets, and grad_output
    y_pred = tl.load(y_pred_ptr + offsets, mask=offsets < n_elements, other=0.0)
    y_true = tl.load(y_true_ptr + offsets, mask=offsets < n_elements, other=0.0)
    grad_output = tl.load(grad_output_ptr + offsets, mask=offsets < n_elements, other=0.0)

    # Compute gradient
    if log_target:
        grad_input = -y_true * grad_output
    else:
        grad_input = -y_true / y_pred * grad_output

    # Store the gradient
    tl.store(grad_input_ptr + offsets, grad_input, mask=offsets < n_elements)

def kldiv_backward_triton(y_pred, y_true, grad_output, log_target=False, BLOCK_SIZE=256):
    # Determine the number of elements and configure execution
    n_elements = y_pred.numel()
    num_warps = get_num_warps(BLOCK_SIZE)

    # Allocate gradient input
    grad_input = torch.empty_like(y_pred)

    # Launch the kernel
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    _kldiv_kernel_backward[grid](y_pred, y_true, grad_output, grad_input, n_elements, log_target, BLOCK_SIZE=BLOCK_SIZE)

    return grad_input
