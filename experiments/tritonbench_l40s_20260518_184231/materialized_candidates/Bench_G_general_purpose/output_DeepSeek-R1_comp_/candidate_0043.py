import torch
import triton
import triton.language as tl

# Forward Kernel
@triton.jit
def _kldiv_kernel_forward(
    y_pred_ptr,
    y_true_ptr,
    loss_ptr,
    loss_sum_ptr,
    BT,
    V,
    log_target: tl.constexpr,
    eps: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    REDUCE_NONE = 0
    REDUCE_SUM = 1
    REDUCE_MEAN = 2
    REDUCE_BATCHMEAN = 3

    pid_bt = tl.program_id(0)
    pid_col_block = tl.program_id(1)

    if pid_bt >= BT:
        return

    col_start = pid_col_block * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < V

    y_pred = tl.load(y_pred_ptr + pid_bt * V + col_offsets, mask=mask, other=0.0)
    y_true = tl.load(y_true_ptr + pid_bt * V + col_offsets, mask=mask, other=0.0)

    if log_target:
        y_true_exp = tl.exp(y_true)
        loss = y_true_exp * (y_true - y_pred)
    else:
        y_true_safe = y_true + eps
        log_y_true = tl.log(y_true_safe)
        loss = y_true * (log_y_true - y_pred)

    if reduction == REDUCE_NONE:
        tl.store(loss_ptr + pid_bt * V + col_offsets, loss, mask=mask)
    else:
        sum_loss = tl.sum(loss)
        tl.atomic_add(loss_sum_ptr + pid_bt, sum_loss)

# Backward Kernel
@triton.jit
def _kldiv_kernel_backward(
    target_ptr,
    grad_output_ptr,
    new_grads_ptr,
    BT,
    V,
    log_target: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_col_block = tl.program_id(1)

    if pid_bt >= BT:
        return

    col_start = pid_col_block * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < V

    target = tl.load(target_ptr + pid_bt * V + col_offsets, mask=mask, other=0.0)
    grad_output = tl.load(grad_output_ptr + pid_bt * V + col_offsets, mask=mask, other=0.0)

    if log_target:
        grad = -tl.exp(target) * grad_output
    else:
        grad = -target * grad_output

    tl.store(new_grads_ptr + pid_bt * V + col_offsets, grad, mask=mask)

# Forward Wrapper
def kldiv_forward_triton(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    log_target: bool = False,
    reduction: str = "mean",
    eps: float = 1e-12,
):
    assert reduction in ["none", "sum", "mean", "batchmean"]
    REDUCTION_MODE = {
        "none": 0,
        "sum": 1,
        "mean": 2,
        "batchmean": 3,
    }
    reduction_code = REDUCTION_MODE[reduction]

    BT, V = y_pred.shape
    device = y_pred.device

    if reduction == "none":
        loss = torch.empty_like(y_pred)
        loss_sum = None
    else:
        loss = torch.empty(0, device=device)
        loss_sum = torch.zeros(BT, device=device, dtype=y_pred.dtype)

    BLOCK_SIZE = 1024
    num_col_blocks = triton.cdiv(V, BLOCK_SIZE)
    grid = (BT, num_col_blocks)

    _kldiv_kernel_forward[grid](
        y_pred,
        y_true,
        loss,
        loss_sum,
        BT,
        V,
        log_target,
        eps,
        reduction_code,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    if reduction == "none":
        return loss
    else:
        total = loss_sum.sum()
        if reduction == "sum":
            return total
        elif reduction == "mean":
            return total / (BT * V)
        elif reduction == "batchmean":
            return total / BT
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

# Backward Wrapper
def kldiv_backward_triton(
    target: torch.Tensor,
    grad_output: torch.Tensor,
    new_grads: torch.Tensor,
    log_target: bool = False,
):
    BT, V = target.shape
    grad_output = grad_output.expand_as(target)

    BLOCK_SIZE = 1024
    num_col_blocks = triton.cdiv(V, BLOCK_SIZE)
    grid = (BT, num_col_blocks)

    _kldiv_kernel_backward[grid](
        target,
        grad_output,
        new_grads,
        BT,
        V,
        log_target,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
