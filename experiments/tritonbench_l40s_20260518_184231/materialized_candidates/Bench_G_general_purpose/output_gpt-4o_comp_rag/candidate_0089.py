import triton
import triton.language as tl
import torch

MAX_FUSED_SIZE = 65536 // 4

# Reduction mode constants
_REDUCTION_MODE_NONE = 0
_REDUCTION_MODE_SUM = 1
_REDUCTION_MODE_MEAN = 2
_REDUCTION_MODE_BATCHMEAN = 3

# Mapping string to reduction mode
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
    y_ptr, y_row_stride,
    gt_ptr, gt_row_stride,
    loss_ptr, loss_row_stride,
    n_cols, reduction_mode,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y = tl.load(y_ptr + row_idx * y_row_stride + col_offsets, mask=mask, other=0.0)
    gt = tl.load(gt_ptr + row_idx * gt_row_stride + col_offsets, mask=mask, other=0.0)

    kl_div = gt * (tl.log(gt) - tl.log(y))
    tl.store(loss_ptr + row_idx * loss_row_stride + col_offsets, kl_div, mask=mask)

    if reduction_mode == _REDUCTION_MODE_SUM:
        tl.atomic_add(loss_ptr, tl.sum(kl_div, axis=0))
    elif reduction_mode == _REDUCTION_MODE_MEAN:
        tl.atomic_add(loss_ptr, tl.sum(kl_div, axis=0) / n_cols)
    elif reduction_mode == _REDUCTION_MODE_BATCHMEAN:
        tl.atomic_add(loss_ptr, tl.sum(kl_div, axis=0) / tl.num_programs())

@triton.jit
def _kldiv_kernel_backward(
    y_ptr, y_row_stride,
    gt_ptr, gt_row_stride,
    grad_ptr, grad_row_stride,
    grad_output_ptr, grad_output_row_stride,
    n_cols, reduction_mode,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y = tl.load(y_ptr + row_idx * y_row_stride + col_offsets, mask=mask, other=0.0)
    gt = tl.load(gt_ptr + row_idx * gt_row_stride + col_offsets, mask=mask, other=0.0)
    grad_output = tl.load(grad_output_ptr + row_idx * grad_output_row_stride + col_offsets, mask=mask, other=0.0)

    grad = (y - gt) / y
    tl.store(grad_ptr + row_idx * grad_row_stride + col_offsets, grad * grad_output, mask=mask)

def kldiv_forward_triton(y, gt, reduction='mean'):
    reduction_mode = _str_to_reduction_mode[reduction]
    n_rows, n_cols = y.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    loss = torch.empty_like(y)
    _kldiv_kernel_forward[(n_rows,)](
        y, y.stride(0),
        gt, gt.stride(0),
        loss, loss.stride(0),
        n_cols, reduction_mode,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    if reduction_mode == _REDUCTION_MODE_SUM:
        return loss.sum()
    elif reduction_mode == _REDUCTION_MODE_MEAN:
        return loss.mean()
    elif reduction_mode == _REDUCTION_MODE_BATCHMEAN:
        return loss.sum() / n_rows
    else:
        return loss

def kldiv_backward_triton(y, gt, grad_output, reduction='mean'):
    reduction_mode = _str_to_reduction_mode[reduction]
    n_rows, n_cols = y.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    grad = torch.empty_like(y)
    _kldiv_kernel_backward[(n_rows,)](
        y, y.stride(0),
        gt, gt.stride(0),
        grad, grad.stride(0),
        grad_output, grad_output.stride(0),
        n_cols, reduction_mode,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    return grad

def calculate_settings(n):
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = get_num_warps(BLOCK_SIZE)
    return BLOCK_SIZE, num_warps
