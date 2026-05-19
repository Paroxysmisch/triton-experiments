import torch
import triton
import triton.language as tl

# Constants
MAX_FUSED_SIZE = 65536 // 4  # 16384
_REDUCTION_MODE_NONE = 0
_REDUCTION_MODE_SUM = 1
_REDUCTION_MODE_MEAN = 2
_REDUCTION_MODE_BATCHMEAN = 3
_str_to_reduction_mode = {
    'none': _REDUCTION_MODE_NONE,
    'sum': _REDUCTION_MODE_SUM,
    'mean': _REDUCTION_MODE_MEAN,
    'batchmean': _REDUCTION_MODE_BATCHMEAN
}

def get_num_warps(BLOCK_SIZE):
    if BLOCK_SIZE >= 32768:
        return 32
    elif BLOCK_SIZE >= 8192:
        return 16
    elif BLOCK_SIZE >= 2048:
        return 8
    else:
        return 4

@triton.jit
def _kldiv_kernel_forward(
    y_ptr, gt_ptr, loss_ptr,
    y_stride, gt_stride, loss_stride,
    reduction,
    total_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Load data
    y = tl.load(y_ptr + offsets * y_stride, mask=mask)
    gt = tl.load(gt_ptr + offsets * gt_stride, mask=mask)

    # Compute log terms safely (avoid log(0))
    log_y = tl.where(y > 0, tl.log(y), 0.0)
    log_gt = tl.where(gt > 0, tl.log(gt), 0.0)
    terms = gt * (log_gt - log_y)

    if reduction == _REDUCTION_MODE_NONE:
        tl.store(loss_ptr + offsets * loss_stride, terms, mask=mask)
    else:
        # Sum terms in block and accumulate
        block_sum = tl.sum(terms, axis=0)
        tl.atomic_add(loss_ptr, block_sum.to(tl.float32))

@triton.jit
def _kldiv_kernel_backward(
    y_ptr, gt_ptr, grad_output_ptr, grad_input_ptr,
    y_stride, gt_stride, grad_output_stride, grad_input_stride,
    reduction,
    total_elements,
    log_target: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Load data
    y = tl.load(y_ptr + offsets * y_stride, mask=mask)
    gt = tl.load(gt_ptr + offsets * gt_stride, mask=mask)

    # Compute gradient components
    if log_target:
        gt_val = tl.exp(gt)
        grad = -gt_val / y
    else:
        grad = -gt / y

    # Load grad_output
    if reduction == _REDUCTION_MODE_NONE:
        go = tl.load(grad_output_ptr + offsets * grad_output_stride, mask=mask)
    else:
        go = tl.load(grad_output_ptr)

    # Apply grad_output scaling
    grad *= go
    tl.store(grad_input_ptr + offsets * grad_input_stride, grad, mask=mask)

def kldiv_forward_triton(y: torch.Tensor, gt: torch.Tensor, reduction: str = 'mean'):
    assert y.is_cuda and gt.is_cuda, "Inputs must be CUDA tensors"
    assert y.shape == gt.shape, "Input shapes must match"

    reduction_mode = _str_to_reduction_mode[reduction]
    total_elements = y.numel()
    batch_size = y.shape[0] if y.dim() > 0 else 1

    # Allocate output tensor
    if reduction == 'none':
        loss = torch.empty_like(y)
    else:
        loss = torch.zeros(1, device=y.device)

    # Configure kernel
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(total_elements))
    if BLOCK_SIZE < 128:
        BLOCK_SIZE = 128
    num_warps = get_num_warps(BLOCK_SIZE)
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    # Launch kernel
    _kldiv_kernel_forward[grid](
        y, gt, loss,
        y.stride(0), gt.stride(0), loss.stride(0),
        reduction_mode,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    # Apply final reduction scaling
    if reduction == 'mean':
        loss = loss / total_elements
    elif reduction == 'batchmean':
        loss = loss / batch_size
    return loss

def kldiv_backward_triton(
    grad_output: torch.Tensor,
    y: torch.Tensor,
    gt: torch.Tensor,
    log_target: bool = False,
    reduction: str = 'mean'
):
    assert y.is_cuda and gt.is_cuda and grad_output.is_cuda, "Inputs must be CUDA tensors"
    assert y.shape == gt.shape, "Input shapes must match"

    reduction_mode = _str_to_reduction_mode[reduction]
    total_elements = y.numel()

    # Allocate gradient tensor
    grad_input = torch.empty_like(y)

    # Configure kernel
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(total_elements))
    if BLOCK_SIZE < 128:
        BLOCK_SIZE = 128
    num_warps = get_num_warps(BLOCK_SIZE)
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    # Launch kernel
    _kldiv_kernel_backward[grid](
        y, gt, grad_output, grad_input,
        y.stride(0), gt.stride(0), grad_output.stride(0), grad_input.stride(0),
        reduction_mode,
        total_elements,
        log_target,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    return grad_input
