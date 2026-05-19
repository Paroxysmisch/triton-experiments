import triton
import triton.language as tl
import torch

# Reduction modes
_REDUCTION_MODE_NONE = 'none'
_REDUCTION_MODE_SUM = 'sum'
_REDUCTION_MODE_MEAN = 'mean'
_REDUCTION_MODE_BATCHMEAN = 'batchmean'

# Helper function to determine the number of warps
def get_num_warps(block_size):
    if block_size >= 32768:
        return 32
    elif block_size >= 8192:
        return 16
    elif block_size >= 2048:
        return 8
    else:
        return 4

@triton.jit
def _kldiv_kernel_forward(
    y_pred, y_pred_stride,
    y_true, y_true_stride,
    output, output_stride,
    n_cols, log_target,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y_pred_row = tl.load(y_pred + row_idx * y_pred_stride + col_offsets, mask=mask, other=0.0)
    y_true_row = tl.load(y_true + row_idx * y_true_stride + col_offsets, mask=mask, other=0.0)

    if log_target:
        kl_div = tl.exp(y_true_row) * (y_true_row - y_pred_row)
    else:
        kl_div = y_true_row * (tl.log(y_true_row) - y_pred_row)

    tl.store(output + row_idx * output_stride + col_offsets, kl_div, mask=mask)

@triton.jit
def _kldiv_kernel_backward(
    grad_output, grad_output_stride,
    y_pred, y_pred_stride,
    y_true, y_true_stride,
    grad_input, grad_input_stride,
    n_cols, log_target,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y_pred_row = tl.load(y_pred + row_idx * y_pred_stride + col_offsets, mask=mask, other=0.0)
    y_true_row = tl.load(y_true + row_idx * y_true_stride + col_offsets, mask=mask, other=0.0)
    grad_output_row = tl.load(grad_output + row_idx * grad_output_stride + col_offsets, mask=mask, other=0.0)

    if log_target:
        grad = -tl.exp(y_true_row)
    else:
        grad = -y_true_row / y_pred_row

    grad_input_row = grad * grad_output_row
    tl.store(grad_input + row_idx * grad_input_stride + col_offsets, grad_input_row, mask=mask)

class KLDivFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_pred, y_true, log_target, reduction):
        n_rows, n_cols = y_pred.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        output = torch.empty_like(y_pred)

        _kldiv_kernel_forward[(n_rows,)](
            y_pred, y_pred.stride(0),
            y_true, y_true.stride(0),
            output, output.stride(0),
            n_cols, log_target,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )

        ctx.save_for_backward(y_pred, y_true)
        ctx.log_target = log_target
        ctx.reduction = reduction

        if reduction == _REDUCTION_MODE_SUM:
            return output.sum()
        elif reduction == _REDUCTION_MODE_MEAN:
            return output.mean()
        elif reduction == _REDUCTION_MODE_BATCHMEAN:
            return output.sum() / n_rows
        else:
            return output

    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y_true = ctx.saved_tensors
        n_rows, n_cols = y_pred.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        grad_input = torch.empty_like(y_pred)

        _kldiv_kernel_backward[(n_rows,)](
            grad_output, grad_output.stride(0),
            y_pred, y_pred.stride(0),
            y_true, y_true.stride(0),
            grad_input, grad_input.stride(0),
            n_cols, ctx.log_target,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )

        if ctx.reduction == _REDUCTION_MODE_MEAN:
            grad_input /= n_rows * n_cols
        elif ctx.reduction == _REDUCTION_MODE_BATCHMEAN:
            grad_input /= n_rows

        return grad_input, None, None, None

def kldiv_forward_triton(y_pred, y_true, log_target=False, reduction='none'):
    return KLDivFunction.apply(y_pred, y_true, log_target, reduction)

def kldiv_backward_triton(grad_output, y_pred, y_true, log_target=False):
    return KLDivFunction.apply(grad_output, y_pred, y_true, log_target)
