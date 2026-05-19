import torch
import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    lse_ptr,
    loss_ptr,
    logits_ptr,
    logits_row_stride,
    labels_ptr,
    lse_square_scale,
    smoothing,
    smoothing_value,
    ignored_index,
    logit_scale,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    LABEL_SMOOTHING: tl.constexpr,
    DISTRIBUTED: tl.constexpr,
    N_DEVICES: tl.constexpr,
):
    row_block_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_offset = row_block_idx * logits_row_stride + col_block_idx * BLOCK_SIZE
    lse_offset = row_block_idx * BLOCK_SIZE + col_block_idx * BLOCK_SIZE * n_rows

    logits_max = -float("inf")
    logits_sum = 0.0
    lse = 0.0
    loss = 0.0

    for i in range(0, BLOCK_SIZE):
        for j in range(0, BLOCK_SIZE):
            logits_idx = logits_offset + i * n_cols + j
            label_idx = col_block_idx * BLOCK_SIZE + i

            label = tl.load(labels_ptr + label_idx)
            logits_ij = tl.load(logits_ptr + logits_idx) / logit_scale

            if LABEL_SMOOTHING:
                if label != ignored_index:
                    one_hot = label == (col_block_idx * BLOCK_SIZE + j)
                    logits_ij = tl.where(one_hot, logits_ij * (1 - smoothing) + smoothing_value, logits_ij * (1 - smoothing))

            logits_max_new = tl.maximum(logits_max, logits_ij)
            logits_sum += logits_ij - logits_max
            logits_max = logits_max_new

    lse = logits_max + tl.log(tl.sum(tl.exp(logits_sum - logits_max), axis=0))

    for i in range(0, BLOCK_SIZE):
        logits_idx = logits_offset + i * n_cols
        label_idx = col_block_idx * BLOCK_SIZE + i

        label = tl.load(labels_ptr + label_idx)

        if label != ignored_index:
            loss += lse
            if LABEL_SMOOTHING:
                logits_ij = tl.load(logits_ptr + logits_idx + n_cols - 1) / logit_scale
                if (col_block_idx * BLOCK_SIZE + i) == label:
                    loss -= logits_ij

    tl.store(lse_ptr + lse_offset, lse + lse_square_scale * lse * lse)
    tl.store(loss_ptr + logits_offset, loss)

@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr,
    dloss_ptr,
    logits_ptr,
    logits_row_stride,
    labels_ptr,
    lse_ptr,
    lse_square_scale,
    smoothing,
    smoothing_value,
    ignored_index,
    logit_scale,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    LABEL_SMOOTHING: tl.constexpr,
    DISTRIBUTED: tl.constexpr,
    N_DEVICES: tl.constexpr,
):
    row_block_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_offset = row_block_idx * logits_row_stride + col_block_idx * BLOCK_SIZE
    lse_offset = row_block_idx * BLOCK_SIZE + col_block_idx * BLOCK_SIZE * n_rows

    dlogits = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    dloss = 0.0

    for i in range(0, BLOCK_SIZE):
        for j in range(0, BLOCK_SIZE):
            logits_idx = logits_offset + i * n_cols + j
            label_idx = col_block_idx * BLOCK_SIZE + i

            label = tl.load(labels_ptr + label_idx)
            logits_ij = tl.load(logits_ptr + logits_idx) / logit_scale
            lse = tl.load(lse_ptr + lse_offset)

            if LABEL_SMOOTHING:
                if label != ignored_index:
                    one_hot = label == (col_block_idx * BLOCK_SIZE + j)
                    dlogits_ij = tl.where(one_hot, 1 - smoothing, 0 - smoothing)
                    logits_ij = tl.where(one_hot, logits_ij * (1 - smoothing) + smoothing_value, logits_ij * (1 - smoothing))
                else:
                    dlogits_ij = 0
                    logits_ij = logits_ij * (1 - smoothing)
            else:
                dlogits_ij = tl.where(label == (col_block_idx * BLOCK_SIZE + j), 1, 0)

            dloss_part = tl.exp(logits_ij - lse)
            dloss += dloss_part
            dlogits[i, j] += dloss_part

            if LABEL_SMOOTHING:
                if (col_block_idx * BLOCK_SIZE + i) == label:
                    dloss -= tl.exp(logits_ij)

    dloss += tl.load(dloss_ptr + logits_offset)

    for i in range(0, BLOCK_SIZE):
        for j in range(0, BLOCK_SIZE):
            logits_idx = logits_offset + i * n_cols + j
            label_idx = col_block_idx * BLOCK_SIZE + i

            label = tl.load(labels_ptr + label_idx)

            if LABEL_SMOOTHING and label != ignored_index:
                dlogits_smoothed = tl.load(dlogits_ptr + logits_idx) / logit_scale
                dlogits_ij = dlogits[i, j] * (1 - smoothing) + dloss * dlogits_smoothed * smoothing
                tl.store(dlogits_ptr + logits_idx, dlogits_ij * logit_scale)
            else:
                dlogits_ij = dlogits[i, j] * dloss
                tl.store(dlogits_ptr + logits_idx, dlogits_ij * logit_scale)

class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        labels,
        smoothing,
        lse_square_scale,
        ignored_index=-100,
        process_group=None,
        logit_scale=1.0,
    ):
        if process_group is not None:
            from ignite.utils import manual_seed

            manual_seed(1234)
            world_size = len(process_group)
            rank = process_group.rank()
            n_rows = (logits.shape[0] + world_size - 1) // world_size
            logits = logits[rank * n_rows : (rank + 1) * n_rows]
            labels = labels[rank * n_rows : (rank + 1) * n
