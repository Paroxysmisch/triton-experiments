import torch
import triton
import triton.language as tl

@triton.jit
def _kldiv_kernel_forward(
    y_pred_ptr, y_true_ptr, output_ptr,
    stride_bt, stride_v,
    BT, V,
    log_target: tl.constexpr, reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BT * V

    y_pred = tl.load(y_pred_ptr + offsets * stride_v, mask=mask)
    y_true = tl.load(y_true_ptr + offsets * stride_v, mask=mask)

    if log_target:
        kl = tl.exp(y_true) * (y_true - y_pred)
    else:
        kl = y_true * (tl.log(y_true + 1e-8) - y_pred)

    if reduction == "none":
        tl.store(output_ptr + offsets * stride_v, kl, mask=mask)
    else:
        tl.atomic_add(output_ptr, tl.sum(kl, axis=0))

@triton.jit
def _kldiv_kernel_backward(
    grad_output_ptr, target_ptr, new_grads_ptr,
    stride_bt, stride_v,
    BT, V,
    log_target: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BT * V

    grad_output = tl.load(grad_output_ptr + offsets * stride_v, mask=mask)
    target = tl.load(target_ptr + offsets * stride_v, mask=mask)

    if log_target:
        grad = -tl.exp(target) * grad_output
    else:
        grad = -target * grad_output

    tl.store(new_grads_ptr + offsets * stride_v, grad, mask=mask)

def kldiv_forward_triton(y_pred, y_true, log_target=False, reduction='mean', eps=1e-8):
    BT, V = y_pred.shape
    output = torch.empty_like(y_pred) if reduction == 'none' else torch.zeros(1, device=y_pred.device, dtype=y_pred.dtype)

    BLOCK_SIZE = triton.next_power_of_2(min(BT * V, 2048))
    num_warps = 4

    grid = lambda meta: (triton.cdiv(BT * V, meta['BLOCK_SIZE']),)

    _kldiv_kernel_forward[grid](
        y_pred, y_true, output,
        y_pred.stride(0), y_pred.stride(1),
        BT, V,
        log_target, reduction,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    if reduction == 'mean':
        output /= (BT * V)
    elif reduction == 'batchmean':
        output /= BT

    return output

def kldiv_backward_triton(target, grad_output, log_target=False):
    BT, V = target.shape
    new_grads = torch.empty_like(target)

    BLOCK_SIZE = triton.next_power_of_2(min(BT * V, 2048))
    num_warps = 4

    grid = lambda meta: (triton.cdiv(BT * V, meta['BLOCK_SIZE']),)

    _kldiv_kernel_backward[grid](
        grad_output, target, new_grads,
        target.stride(0), target.stride(1),
        BT, V,
        log_target,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    return new_grads

class KLDivLossTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_pred, y_true, log_target=False, reduction='mean', eps=1e-8):
        ctx.save_for_backward(y_true)
        ctx.log_target = log_target
        ctx.reduction = reduction
        return kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps)

    @staticmethod
    def backward(ctx, grad_output):
        y_true, = ctx.saved_tensors
        return kldiv_backward_triton(y_true, grad_output, ctx.log_target), None, None, None, None

def kl_div_loss_triton(y_pred, y_true, log_target=False, reduction='mean', eps=1e-8):
    return KLDivLossTriton.apply(y_pred, y_true, log_target, reduction, eps)
