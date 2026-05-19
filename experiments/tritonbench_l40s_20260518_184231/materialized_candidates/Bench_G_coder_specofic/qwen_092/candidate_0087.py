import triton
import triton.language as tl

# Constants
MAX_FUSED_SIZE = 65536 // 4
_REDUCTION_MODE_NONE = 0
_REDUCTION_MODE_SUM = 1
_REDUCTION_MODE_MEAN = 2
_REDUCTION_MODE_BATCHMEAN = 3
_str_to_reduction_mode = {
    "none": _REDUCTION_MODE_NONE,
    "sum": _REDUCTION_MODE_SUM,
    "mean": _REDUCTION_MODE_MEAN,
    "batchmean": _REDUCTION_MODE_BATCHMEAN
}

# Helper function to determine the number of warps
@triton.jit
def get_num_warps(BLOCK_SIZE):
    return 4
    # Increase the number of warps for larger block sizes
    if BLOCK_SIZE >= 32768:
        return 32

# Forward kernel
@triton.jit
def _kldiv_kernel_forward(
    y_ptr, gt_ptr, loss_ptr,
    y_stride, gt_stride, loss_stride,
    BLOCK_SIZE: tl.constexpr, REDUCTION_MODE: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    row_offset = row * BLOCK_SIZE
    col_offset = col * BLOCK_SIZE
    row_end = row_offset + BLOCK_SIZE
    col_end = col_offset + BLOCK_SIZE

    # Initialize loss tensor
    loss = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for i in range(row_offset, row_end):
        for j in range(col_offset, col_end):
            y_val = y_ptr[i * y_stride + j]
            gt_val = gt_ptr[i * gt_stride + j]
            if gt_val > 0 and y_val > 0:
                loss[j - col_offset] += gt_val * (tl.log(gt_val) - tl.log(y_val))

    # Reduction
    if REDUCTION_MODE == _REDUCTION_MODE_SUM:
        tl.store(loss_ptr + row * loss_stride, tl.sum(loss))
    elif REDUCTION_MODE == _REDUCTION_MODE_MEAN:
        tl.store(loss_ptr + row * loss_stride, tl.sum(loss) / BLOCK_SIZE)
    elif REDUCTION_MODE == _REDUCTION_MODE_BATCHMEAN:
        tl.store(loss_ptr + row * loss_stride, tl.sum(loss) / (BLOCK_SIZE * BLOCK_SIZE))

# Backward kernel
@triton.jit
def _kldiv_kernel_backward(
    y_ptr, gt_ptr, grad_ptr,
    y_stride, gt_stride, grad_stride,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    row_offset = row * BLOCK_SIZE
    col_offset = col * BLOCK_SIZE
    row_end = row_offset + BLOCK_SIZE
    col_end = col_offset + BLOCK_SIZE

    # Initialize grad tensor
    grad = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for i in range(row_offset, row_end):
        for j in range(col_offset, col_end):
            y_val = y_ptr[i * y_stride + j]
            gt_val = gt_ptr[i * gt_stride + j]
            if gt_val > 0 and y_val > 0:
                grad[j - col_offset] += gt_val / y_val

    # Store gradients
    for j in range(col_offset, col_end):
        tl.store(grad_ptr + (row * BLOCK_SIZE + j) * grad_stride, grad[j - col_offset])

# Forward wrapper function
@triton.jit
def kldiv_forward_triton(
    y, gt, loss,
    y_shape, gt_shape, loss_shape,
    reduction: str,
    BLOCK_SIZE: tl.constexpr
):
    REDUCTION_MODE = _str_to_reduction_mode.get(reduction, _REDUCTION_MODE_NONE)
    grid = (y_shape[0], (y_shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE)
    num_warps = get_num_warps(BLOCK_SIZE)
    _kldiv_kernel_forward[grid, (BLOCK_SIZE, num_warps)](
        y, gt, loss,
        y_shape[1], gt_shape[1], loss_shape[1],
        BLOCK_SIZE, REDUCTION_MODE
    )

# Backward wrapper function
@triton.jit
def kldiv_backward_triton(
    y, gt, grad_output, grad,
    y_shape, gt_shape, grad_shape,
    BLOCK_SIZE: tl.constexpr
):
    grid = (y_shape[0], (y_shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE)
    num_warps = get_num_warps(BLOCK_SIZE)
    _kldiv_kernel_backward[grid, (BLOCK_SIZE, num_warps)](
        y, gt, grad,
        y_shape[1], gt_shape[1], grad_shape[1],
        BLOCK_SIZE
    )
    if grad_output.shape == (1,) and tl.load(grad_output) == 1.0:
        return grad
    else:
        return grad * tl.load(grad_output)
