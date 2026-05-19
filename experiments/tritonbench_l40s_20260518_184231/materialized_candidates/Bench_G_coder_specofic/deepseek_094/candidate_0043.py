import triton
import triton.language as tl

@triton.jit
def _kldiv_kernel_forward(
    y_pred_p,
    y_true_p,
    log_target_p,
    loss_ptr_p,
    BLOCK_SIZE: tl.constexpr,
):
    # Define the block indices
    block_start = tl.program_id(0) * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE

    # Load data from global memory to shared memory
    y_pred = tl.load(y_pred_p + block_start, mask=block_end > y_pred_p.shape[0])
    y_true = tl.load(y_true_p + block_start, mask=block_end > y_true_p.shape[0])
    log_target = tl.load(log_target_p + block_start, mask=block_end > log_target_p.shape[0])

    # Compute KL divergence
    if not log_target:
        loss = y_true * (tl.log(y_true) - y_pred)
    else:
        loss = tl.exp(y_true) * (y_true - y_pred)

    # Reduce loss across block
    loss = tl.sum(loss, axis=0)

    # Store loss in shared memory
    tl.store(loss_ptr_p + block_start, loss)

def kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps, BLOCK_SIZE=64):
    # Allocate memory for loss
    loss = tl.zeros((y_pred.shape[0],), dtype=tl.float32)

    # Run kernel
    _kldiv_kernel_forward[((y_pred.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE, )](
        y_pred,
        y_true,
        log_target,
        loss,
        BLOCK_SIZE,
    )

    # Reduce loss across batch
    if reduction == "sum":
        loss = tl.sum(loss)
    elif reduction == "mean":
        loss = tl.sum(loss) / y_pred.shape[0]
    elif reduction == "batchmean":
        loss = tl.sum(loss) / y_pred.shape[1]

    return loss
