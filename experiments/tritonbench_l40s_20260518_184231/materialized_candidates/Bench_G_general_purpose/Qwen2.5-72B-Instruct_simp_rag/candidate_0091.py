import triton
import triton.language as tl
import torch

MAX_FUSED_SIZE = 65536
next_power_of_2 = triton.next_power_of_2

def calculate_settings(n):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps

_REDUCTION_MODE_NONE = 0
_REDUCTION_MODE_SUM = 1
_REDUCTION_MODE_MEAN = 2
_REDUCTION_MODE_BATCHMEAN = 3

@triton.jit
def _kldiv_kernel_forward(
    y_pred, y_pred_row_stride,
    y_true, y_true_row_stride,
    output, output_row_stride,
    n_cols, log_target, reduction,
    BLOCK_SIZE : tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y_pred += row_idx * y_pred_row_stride
    y_true += row_idx * y_true_row_stride
    output += row_idx * output_row_stride

    y_pred_row = tl.load(y_pred + col_offsets, mask=mask, other=0).to(tl.float32)
    y_true_row = tl.load(y_true + col_offsets, mask=mask, other=0).to(tl.float32)

    if log_target:
        y_true_row = tl.exp(y_true_row)

    kl_div = y_true_row * (tl.log(y_true_row) - y_pred_row)
    if reduction == _REDUCTION_MODE_NONE:
        tl.store(output + col_offsets, kl_div, mask=mask)
    elif reduction == _REDUCTION_MODE_SUM:
        sum_kl_div = tl.sum(kl_div, axis=0)
        tl.store(output + row_idx, sum_kl_div)
    elif reduction == _REDUCTION_MODE_MEAN:
        mean_kl_div = tl.sum(kl_div, axis=0) / n_cols
        tl.store(output + row_idx, mean_kl_div)
    elif reduction == _REDUCTION_MODE_BATCHMEAN:
        mean_kl_div = tl.sum(kl_div, axis=0) / n_cols
        tl.atomic_add(output, mean_kl_div)

def kldiv_forward_triton(y_pred, y_true, log_target, reduction):
    shape = y_pred.shape
    dim = shape[-1]
    y_pred = y_pred.view(-1, dim)
    y_true = y_true.view(-1, dim)
    n_rows, n_cols = y_pred.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    output = torch.empty((n_rows, n_cols) if reduction == 'none' else (n_rows, 1), dtype=y_pred.dtype, device="cuda")

    _kldiv_kernel_forward[(n_rows,)](
        y_pred, y_pred.stride(0),
        y_true, y_true.stride(0),
        output, output.stride(0),
        n_cols, log_target, 
        _REDUCTION_MODE_NONE if reduction == 'none' else 
        _REDUCTION_MODE_SUM if reduction == 'sum' else 
        _REDUCTION_MODE_MEAN if reduction == 'mean' else 
        _REDUCTION_MODE_BATCHMEAN,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return output.view(*shape) if reduction == 'none' else output.sum() if reduction == 'sum' else output.mean() if reduction == 'mean' else output

@triton.jit
def _kldiv_kernel_backward(
    y_pred, y_pred_row_stride,
    y_true, y_true_row_stride,
    grad_output, grad_output_row_stride,
    grad_y_pred, grad_y_pred_row_stride,
    n_cols, log_target,
    BLOCK_SIZE : tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    y_pred += row_idx * y_pred_row_stride
    y_true += row_idx * y_true_row_stride
    grad_output += row_idx * grad_output_row_stride
    grad_y_pred += row_idx * grad_y_pred_row_stride

    y_pred_row = tl.load(y_pred + col_offsets, mask=mask, other=0).to(tl.float32)
    y_true_row = tl.load(y_true + col_offsets, mask=mask, other=0).to(tl.float32)
    grad_output_row = tl.load(grad_output + col_offsets, mask=mask, other=0).to(tl.float32)

    if log_target:
        y_true_row = tl.exp(y_true_row)

    grad_y_pred_row = -y_true_row * grad_output_row
    tl.store(grad_y_pred + col_offsets, grad_y_pred_row, mask=mask)

def kldiv_backward_triton(y_pred, y_true, grad_output, log_target):
    shape = y_pred.shape
    dim = shape[-1]
    y_pred = y_pred.view(-1, dim)
    y_true = y_true.view(-1, dim)
    grad_output = grad_output.view(-1, dim)
    n_rows, n_cols = y_pred.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    grad_y_pred = torch.empty_like(y_pred)

    _kldiv_kernel_backward[(n_rows,)](
        y_pred, y_pred.stride(0),
        y_true, y_true.stride(0),
        grad_output, grad_output.stride(0),
        grad_y_pred, grad_y_pred.stride(0),
        n_cols, log_target,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return grad_y_pred.view(*shape)

class KLDivTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_pred, y_true, log_target, reduction):
        ctx.log_target = log_target
        ctx.reduction = reduction
        ctx.save_for_backward(y_pred, y_true)
        return kldiv_forward_triton(y_pred, y_true, log_target, reduction)

    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y_true = ctx.saved_tensors
        return kldiv_backward_triton(y_pred, y_true, grad_output, ctx.log_target), None, None, None

# Example usage
y_pred = torch.randn(10, 5, device='cuda', requires_grad=True)
y_true = torch.randn(10, 5, device='cuda')

log_target = False
reduction = 'mean'

output = KLDivTriton.apply(y_pred, y_true, log_target, reduction)
output.backward()

print("KL Divergence:", output)
print("Gradient of y_pred:", y_pred.grad)
