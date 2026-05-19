import triton
import triton.language as tl

# Constants
MAX_FUSED_SIZE = 65536 // 4
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

# Function to determine the optimal number of warps
def get_num_warps(BLOCK_SIZE):
    if BLOCK_SIZE >= 32768:
        return 32
    elif BLOCK_SIZE >= 16384:
        return 16
    elif BLOCK_SIZE >= 8192:
        return 8
    elif BLOCK_SIZE >= 4096:
        return 4
    else:
        return 4

# Forward kernel
@triton.jit
def _kldiv_kernel_forward(y_ptr, gt_ptr, loss_ptr, y_stride, gt_stride, loss_stride, n_elements, reduction_mode, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    y = tl.load(y_ptr + offsets * y_stride, mask=mask)
    gt = tl.load(gt_ptr + offsets * gt_stride, mask=mask)

    # Compute KL divergence
    log_y = tl.log(y)
    log_gt = tl.log(gt)
    kl = gt * (log_gt - log_y)

    if reduction_mode == _REDUCTION_MODE_SUM:
        kl_sum = tl.sum(kl, axis=0)
        tl.store(loss_ptr + pid, kl_sum)
    elif reduction_mode == _REDUCTION_MODE_MEAN:
        kl_mean = tl.sum(kl, axis=0) / n_elements
        tl.store(loss_ptr + pid, kl_mean)
    elif reduction_mode == _REDUCTION_MODE_BATCHMEAN:
        kl_batchmean = tl.sum(kl, axis=0) / (n_elements // BLOCK_SIZE)
        tl.store(loss_ptr + pid, kl_batchmean)
    else:
        tl.store(loss_ptr + offsets * loss_stride, kl, mask=mask)

# Backward kernel
@triton.jit
def _kldiv_kernel_backward(grad_output_ptr, y_ptr, gt_ptr, grad_y_ptr, y_stride, gt_stride, grad_y_stride, n_elements, log_target, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    y = tl.load(y_ptr + offsets * y_stride, mask=mask)
    gt = tl.load(gt_ptr + offsets * gt_stride, mask=mask)
    grad_output = tl.load(grad_output_ptr, mask=mask)

    if log_target:
        grad_y = -tl.exp(gt) * grad_output
    else:
        grad_y = -gt * grad_output

    tl.store(grad_y_ptr + offsets * grad_y_stride, grad_y, mask=mask)

# Wrapper function for forward pass
def kldiv_forward_triton(y, gt, reduction='mean'):
    n_elements = y.numel()
    reduction_mode = _str_to_reduction_mode[reduction]
    BLOCK_SIZE = min(n_elements, MAX_FUSED_SIZE)
    num_warps = get_num_warps(BLOCK_SIZE)

    y_ptr = y.data_ptr()
    gt_ptr = gt.data_ptr()
    loss = triton.empty((n_elements,), device=y.device, dtype=y.dtype)
    loss_ptr = loss.data_ptr()

    _kldiv_kernel_forward[(n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, num_warps](
        y_ptr, gt_ptr, loss_ptr, y.stride(0), gt.stride(0), loss.stride(0), n_elements, reduction_mode, BLOCK_SIZE
    )

    if reduction_mode in [_REDUCTION_MODE_SUM, _REDUCTION_MODE_MEAN, _REDUCTION_MODE_BATCHMEAN]:
        return loss.sum()
    else:
        return loss

# Wrapper function for backward pass
def kldiv_backward_triton(grad_output, y, gt, log_target=False):
    n_elements = y.numel()
    BLOCK_SIZE = min(n_elements, MAX_FUSED_SIZE)
    num_warps = get_num_warps(BLOCK_SIZE)

    y_ptr = y.data_ptr()
    gt_ptr = gt.data_ptr()
    grad_y = triton.empty((n_elements,), device=y.device, dtype=y.dtype)
    grad_y_ptr = grad_y.data_ptr()
    grad_output_ptr = grad_output.data_ptr()

    _kldiv_kernel_backward[(n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, num_warps](
        grad_output_ptr, y_ptr, gt_ptr, grad_y_ptr, y.stride(0), gt.stride(0), grad_y.stride(0), n_elements, log_target, BLOCK_SIZE
    )

    if grad_output.numel() == 1 and grad_output.item() == 1.0:
        return grad_y
    else:
        return grad_y * grad_output
