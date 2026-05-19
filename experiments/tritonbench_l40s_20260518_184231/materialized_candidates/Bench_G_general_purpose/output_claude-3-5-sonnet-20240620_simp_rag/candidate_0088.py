import triton
import triton.language as tl
import torch

# Reduction mode constants
_REDUCTION_MODE_NONE = 0
_REDUCTION_MODE_SUM = 1
_REDUCTION_MODE_MEAN = 2
_REDUCTION_MODE_BATCHMEAN = 3

def get_num_warps(BLOCK_SIZE):
    if BLOCK_SIZE >= 2048:
        return 8
    elif BLOCK_SIZE >= 1024:
        return 4
    else:
        return 2

@triton.jit
def _kldiv_kernel_forward(
    output_ptr, y_pred_ptr, y_true_ptr,
    n_elements, log_target,
    reduction_mode,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    pred = tl.load(y_pred_ptr + offsets, mask=mask, other=0.0)
    true = tl.load(y_true_ptr + offsets, mask=mask, other=0.0)

    # Convert target to log space if needed
    if not log_target:
        true = tl.log(true + 1e-10)  # Add epsilon for numerical stability

    # Compute KL divergence: target * (log(target) - pred)
    kl_div = tl.exp(true) * (true - pred)

    # Apply reduction
    if reduction_mode == _REDUCTION_MODE_NONE:
        tl.store(output_ptr + offsets, kl_div, mask=mask)
    else:
        # Sum reduction within block
        block_sum = tl.sum(kl_div, axis=0)
        if pid == 0:
            if reduction_mode == _REDUCTION_MODE_MEAN:
                block_sum = block_sum / n_elements
            elif reduction_mode == _REDUCTION_MODE_BATCHMEAN:
                block_sum = block_sum / (n_elements // true.shape[0])
            tl.store(output_ptr, block_sum)

@triton.jit
def _kldiv_kernel_backward(
    grad_output_ptr, grad_input_ptr,
    y_pred_ptr, y_true_ptr,
    n_elements, log_target,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    pred = tl.load(y_pred_ptr + offsets, mask=mask, other=0.0)
    true = tl.load(y_true_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)

    # Convert target to log space if needed
    if not log_target:
        true = tl.log(true + 1e-10)

    # Compute gradient: -exp(target)
    grad_input = -tl.exp(true) * grad

    # Store gradient
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)

def kldiv_forward_triton(y_pred, y_true, log_target=False, reduction='mean'):
    # Input validation
    assert y_pred.is_cuda and y_true.is_cuda
    assert y_pred.shape == y_true.shape
    
    # Get reduction mode
    reduction_modes = {
        'none': _REDUCTION_MODE_NONE,
        'sum': _REDUCTION_MODE_SUM,
        'mean': _REDUCTION_MODE_MEAN,
        'batchmean': _REDUCTION_MODE_BATCHMEAN
    }
    reduction_mode = reduction_modes[reduction]

    # Configure kernel parameters
    n_elements = y_pred.numel()
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 2048))
    num_warps = get_num_warps(BLOCK_SIZE)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Allocate output
    if reduction == 'none':
        output = torch.empty_like(y_pred)
    else:
        output = torch.empty(1, device=y_pred.device, dtype=y_pred.dtype)

    # Launch kernel
    _kldiv_kernel_forward[grid](
        output, y_pred, y_true,
        n_elements, log_target,
        reduction_mode,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return output

def kldiv_backward_triton(grad_output, y_pred, y_true, log_target=False):
    # Input validation
    assert grad_output.is_cuda and y_pred.is_cuda and y_true.is_cuda
    
    # Configure kernel parameters
    n_elements = y_pred.numel()
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 2048))
    num_warps = get_num_warps(BLOCK_SIZE)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Allocate gradient output
    grad_input = torch.empty_like(y_pred)

    # Launch kernel
    _kldiv_kernel_backward[grid](
        grad_output, grad_input,
        y_pred, y_true,
        n_elements, log_target,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return grad_input
