triton
import triton
import triton.language as tl

BLOCK_SIZE = 128

@triton.jit
def _kldiv_kernel_forward(
    y_pred_ptr, y_true_ptr, log_target_ptr, loss_ptr,
    B, T, V, eps, reduction, pid=triton.program_id(0)
):
    batch_id = pid // (T * V)
    block_id = pid % (T * V)
    row = block_id // V
    col = block_id % V

    y_pred = tl.load(y_pred_ptr + row * V * B + col * B, mask=row < B and col < B)
    y_true = tl.load(y_true_ptr + row * V * B + col * B, mask=row < B and col < B)
    log_target = tl.load(log_target_ptr + row * V * B + col * B, mask=row < B and col < B)

    if log_target:
        loss = tl.exp(y_true) * (y_true - y_pred)
    else:
        loss = y_true * (tl.log(y_true + eps) - tl.log(y_pred + eps))

    if reduction == "none":
        tl.store(loss_ptr + row * V * B + col * B, loss, mask=row < B and col < B)
    elif reduction == "sum":
        tl.store(loss_ptr + row * V * B + col * B, loss, mask=row < B and col < B)
        if col == 0:
            for i in range(1, V):
                tl.store(loss_ptr + row * V * B + i * B, 0, mask=row < B and i < B)
    elif reduction == "mean":
        tl.store(loss_ptr + row * V * B + col * B, loss, mask=row < B and col < B)
        if col == 0:
            for i in range(1, V):
                tl.store(loss_ptr + row * V * B + i * B, 0, mask=row < B and i < B)
    elif reduction == "batchmean":
        tl.store(loss_ptr + row * V * B + col * B, loss, mask=row < B and col < B)
        if col == 0:
            for i in range(1, V):
                tl.store(loss_ptr + row * V * B + i * B, 0, mask=row < B and i < B)
