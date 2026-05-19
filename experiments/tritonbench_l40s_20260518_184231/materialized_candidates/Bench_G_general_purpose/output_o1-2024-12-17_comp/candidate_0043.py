import triton
import triton.language as tl
import torch

# -------------------------------
# Forward Kernel
# -------------------------------
@triton.jit
def _kldiv_kernel_forward(
    y_pred_ptr,      # *f32
    y_true_ptr,      # *f32
    loss_ptr,        # *f32
    N,               # i32: total number of elements (BT * V)
    log_target,      # i32: 1 if log_target=True, else 0
    reduction,       # i32: 0=none, 1=sum
    eps,             # f32
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    y_pred_val = tl.load(y_pred_ptr + offset, mask=mask, other=0.0)
    y_true_val = tl.load(y_true_ptr + offset, mask=mask, other=0.0)

    is_log = log_target == 1
    # Compute the KL divergence
    # if log_target:
    #     loss = exp(y_true) * (y_true - y_pred)
    # else:
    #     loss = y_true * (log(y_true + eps) - y_pred)
    if is_log:
        loss_val = tl.exp(y_true_val) * (y_true_val - y_pred_val)
    else:
        loss_val = y_true_val * (tl.log(y_true_val + eps)) - y_true_val * y_pred_val

    # Store or reduce
    if reduction == 0:
        # "none" => write out directly
        tl.store(loss_ptr + offset, loss_val, mask=mask)
    else:
        # "sum" => partial sum and atomic add
        part_sum = tl.sum(loss_val, axis=0)
        if mask[0]:
            tl.atomic_add(loss_ptr, part_sum)


# -------------------------------
# Forward Wrapper
# -------------------------------
def kldiv_forward_triton(y_pred, y_true, log_target=False, reduction="mean", eps=1e-6):
    """
    y_pred:  [BT, V]  (log probabilities)
    y_true:  [BT, V]  (distribution or log-distribution)
    log_target: bool
    reduction: str in {"none", "sum", "mean", "batchmean"}
    eps: float
    """
    # Flatten inputs for block-wise 1D processing
    shape = y_pred.shape
    BT, V = shape
    y_pred_1d = y_pred.flatten()
    y_true_1d = y_true.flatten()
    N = BT * V

    # Prepare output
    if reduction == "none":
        loss = torch.empty_like(y_pred)
    else:
        loss = torch.zeros(1, dtype=y_pred.dtype, device=y_pred.device)

    # Reduction code: 0 => none, 1 => sum
    if reduction == "none":
        red_code = 0
    else:
        red_code = 1

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = lambda META: ( (N + META["BLOCK_SIZE"] - 1) // META["BLOCK_SIZE"], )
    _kldiv_kernel_forward[grid](
        y
