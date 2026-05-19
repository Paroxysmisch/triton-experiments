import torch
import triton
import triton.language as tl

# Constants for reduction modes
_REDUCTION_MODE_NONE = 0
_REDUCTION_MODE_SUM = 1
_REDUCTION_MODE_MEAN = 2
_REDUCTION_MODE_BATCHMEAN = 3

# Block size for parallel execution
BLOCK_SIZE = 1024

def get_num_warps(n_elements):
    """Determine optimal number of warps based on problem size"""
    if n_elements < 2048:
        return 4
    elif n_elements < 4096:
        return 8
    else:
        return 16

@triton.jit
def _kldiv_kernel_forward(
    pred_ptr, target_ptr, output_ptr, 
    n_elements, log_target, reduction_mode,
    BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for KL divergence forward pass"""
    # Get program ID and compute offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input values using mask
    pred = tl.load(pred_ptr + offsets, mask=mask)
    target = tl.load(target_ptr + offsets, mask=mask)
    
    # Compute KL divergence
    if log_target:
        kl = tl.exp(target) * (target - pred)
    else:
        kl = target * (tl.log(target + 1e-10) - pred)
    
    # Apply reduction if needed
    if reduction_mode == _REDUCTION_MODE_NONE:
        tl.store(output_ptr + offsets, kl, mask=mask)
    else:
        # Use parallel reduction for sum/mean modes
        kl = tl.where(mask, kl, 0.0)
        sum_kl = tl.sum(kl, axis=0)
        if pid == 0:
            if reduction_mode == _REDUCTION_MODE_MEAN:
                sum_kl = sum_kl / n_elements
            elif reduction_mode == _REDUCTION_MODE_BATCHMEAN:
                sum_kl = sum_kl / (n_elements // target.shape[0])
            tl.store(output_ptr, sum_kl)

@triton.jit
def _kldiv_kernel_backward(
    grad_output_ptr, pred_ptr, target_ptr, grad_input_ptr,
    n_elements, log_target,
    BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for KL divergence backward pass"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    # Load inputs
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    pred = tl.load(pred_ptr + offsets, mask=mask)
    target = tl.load(target_ptr + offsets, mask=mask)
    
    # Compute gradient
    if log_target:
        grad = -tl.exp(target) * grad_output
    else:
        grad = -target * grad_output
    
    # Store gradient
    tl.store(grad_input_ptr + offsets, grad, mask=mask)

def kldiv_forward_triton(pred, target, log_target=False, reduction='mean'):
    """
    Forward pass for KL divergence using Triton
    Args:
        pred: Prediction tensor (log-probabilities)
        target: Target tensor
        log_target: Whether target is in log-space
        reduction: Reduction method ('none', 'sum', 'mean', 'batchmean')
    """
    n_elements = pred.numel()
    
    # Map reduction type to constant
    reduction_map = {
        'none': _REDUCTION_MODE_NONE,
        'sum': _REDUCTION_MODE_SUM,
        'mean': _REDUCTION_MODE_MEAN,
        'batchmean': _REDUCTION_MODE_BATCHMEAN
    }
    reduction_mode = reduction_map[reduction]
    
    # Allocate output tensor
    if reduction == 'none':
        output = torch.empty_like(pred)
    else:
        output = torch.empty((), device=pred.device, dtype=pred.dtype)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    num_warps = get_num_warps(n_elements)
    
    _kldiv_kernel_forward[grid](
        pred.data_ptr(), target.data_ptr(), output.data_ptr(),
        n_elements, log_target, reduction_mode,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return output

def kldiv_backward_triton(grad_output, pred, target, log_target=False):
    """
    Backward pass for KL divergence using Triton
    Args:
        grad_output: Gradient from upstream
        pred: Original prediction tensor
        target: Original target tensor
        log_target: Whether target was in log-space
    """
    n_elements = pred.numel()
    grad_input = torch.empty_like(pred)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    num_warps = get_num_warps(n_elements)
    
    _kldiv_kernel_backward[grid](
        grad_output.data_ptr(), pred.data_ptr(), target.data_ptr(),
        grad_input.data_ptr(), n_elements, log_target,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return grad_input
