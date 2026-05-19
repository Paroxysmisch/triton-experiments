import triton
import triton.language as tl

# Forward Pass Kernel
@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr,  # *float32
    labels_ptr,  # *int32
    loss_ptr,    # *float32
    lse_ptr,     # *float32
    z_loss_ptr,  # *float32
    smoothing: tl.float32,
    logit_scale: tl.float32,
    lse_square_scale: tl.float32,
    ignored_index: tl.int32,
    total_classes: tl.int32,
    class_start_idx: tl.int32,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    col_start = class_start_idx

    # Initialize accumulators
    lse = -float('inf')
    z_loss = 0.0
    loss = 0.0

    # Iterate over the block of columns
    for col in range(col_start, col_start + BLOCK_SIZE):
        if col >= total_classes:
            break
        logit = logits_ptr[row_start + col]
        logit_scaled = logit * logit_scale
        lse = tl.max(lse, logit_scaled)
        z_loss += tl.exp(logit_scaled)

    # Compute log-sum-exp (lse)
    lse = lse + tl.log(z_loss)

    # Compute loss
    label = labels_ptr[pid]
    if label != ignored_index:
        if HAS_SMOOTHING:
            smoothed_label = (1.0 - smoothing) * (label == col) + smoothing / total_classes
            loss += -tl.sum(smoothed_label * (logit_scaled - lse))
        else:
            loss += -logit_scaled + lse

    # Write results to output pointers
    tl.store(loss_ptr + pid, loss)
    tl.store(lse_ptr + pid, lse)
    tl.store(z_loss_ptr + pid, z_loss)

# Backward Pass Kernel
@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr,  # *float32
    labels_ptr,  # *int32
    dloss_ptr,   # *float32
    dlogits_ptr, # *float32
    lse_ptr,     # *float32
    z_loss_ptr,  # *float32
    smoothing: tl.float32,
    logit_scale: tl.float32,
    ignored_index: tl.int32,
    total_classes: tl.int32,
    class_start_idx: tl.int32,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    col_start = class_start_idx

    # Load lse and z_loss
    lse = tl.load(lse_ptr + pid)
    z_loss = tl.load(z_loss_ptr + pid)
    dloss = tl.load(dloss_ptr + pid)

    # Initialize dlogits
    dlogits = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over the block of columns
    for col in range(col_start, col_start + BLOCK_SIZE):
        if col >= total_classes:
            break
        logit = logits_ptr[row_start + col]
        logit_scaled = logit * logit_scale
        prob = tl.exp(logit_scaled - lse)
        if HAS_SMOOTHING:
            smoothed_label = (1.0 - smoothing) * (labels_ptr[pid] == col) + smoothing / total_classes
            dlogits[col - col_start] = dloss * (prob - smoothed_label)
        else:
            dlogits[col - col_start] = dloss * (prob - (labels_ptr[pid] == col))

    # Write results to output pointer
    for col in range(col_start, col_start + BLOCK_SIZE):
        if col >= total_classes:
            break
        tl.store(dlogits_ptr + row_start + col, dlogits[col - col_start])

# Wrapper Functions
def cross_entropy_fwd(
    logits,  # *float32
    labels,  # *int32
    smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 1.0,
    ignored_index: int = -100,
    total_classes: int = 1000,
    class_start_idx: int = 0,
    BLOCK_SIZE: int = 128,
    HAS_SMOOTHING: bool = False,
    SPLIT: bool = False
):
    assert logits.shape[0] == labels.shape[0]
    assert logits.shape[1] == total_classes

    loss = triton.empty_like(logits, shape=(logits.shape[0],))
    lse = triton.empty_like(logits, shape=(logits.shape[0],))
    z_loss = triton.empty_like(logits, shape=(logits.shape[0],))

    grid = (logits.shape[0],)
    cross_entropy_fwd_kernel[grid](
        logits, labels, loss, lse, z_loss, smoothing, logit_scale, lse_square_scale, ignored_index, total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING, SPLIT
    )

    return loss, lse, z_loss

def cross_entropy_bwd(
    logits,  # *float32
    labels,  # *int32
    dloss,   # *float32
    lse,     # *float32
    z_loss,  # *float32
    smoothing: float = 0.0,
    logit_scale: float = 1.0,
    ignored_index: int = -100,
    total_classes: int = 1000,
    class_start_idx: int = 0,
    BLOCK_SIZE: int = 128,
    HAS_SMOOTHING: bool = False,
    SPLIT: bool = False
):
    assert logits.shape[0] == labels.shape[0]
    assert logits.shape[1] == total_classes

    dlogits = triton.empty_like(logits)

    grid = (logits.shape[0],)
    cross_entropy_bwd_kernel[grid](
        logits, labels, dloss, dlogits, lse, z_loss, smoothing, logit_scale, ignored_index, total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING, SPLIT
    )

    return dlogits

import triton
import triton.language as tl

# Example usage
logits = triton.random((128, 1000), dtype=tl.float32)
labels = triton.random((128,), dtype=tl.int32, low=0, high=1000)

loss, lse, z_loss = cross_entropy_fwd(logits, labels, smoothing=0.1, logit_scale=1.0, ignored_index=-100, total_classes=1000, class_start_idx=0, BLOCK_SIZE=128, HAS_SMOOTHING=True, SPLIT=False)

dloss = triton.ones((128,), dtype=tl.float32)
dlogits = cross_entropy_bwd(logits, labels, dloss, lse, z_loss, smoothing=0.1, logit_scale=1.0, ignored_index=-100, total_classes=1000, class_start_idx=0, BLOCK_SIZE=128, HAS_SMOOTHING=True, SPLIT=False)
