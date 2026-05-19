import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr, labels_ptr, loss_ptr, lse_ptr, z_loss_ptr,
    num_classes, num_rows, smoothing, logit_scale, BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr, SPLIT: tl.constexpr
):
    # Define the row and column index
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)

    # Load the logits for this row
    logits = tl.load(logits_ptr + row_idx * num_classes + col_idx, mask=col_idx < num_classes, other=-float('inf'))

    # Apply scaling to logits if necessary
    if logit_scale != 1.0:
        logits *= logit_scale

    # Compute log-sum-exp for normalization
    max_logits = tl.max(logits, axis=0)
    logits_exp = tl.exp(logits - max_logits)
    lse = tl.log(tl.sum(logits_exp, axis=0)) + max_logits

    # Store lse
    tl.store(lse_ptr + row_idx, lse)

    # Get the label for this row
    label = tl.load(labels_ptr + row_idx)

    # Compute z_loss and loss
    if label != -1:  # Assuming -1 is the ignored_index
        log_prob = logits[label] - lse
        z_loss = -log_prob
        if HAS_SMOOTHING:
            smooth_loss = -tl.sum(logits_exp / num_classes) * smoothing
            z_loss = (1.0 - smoothing) * z_loss + smooth_loss
        tl.store(loss_ptr + row_idx, z_loss)
        tl.store(z_loss_ptr + row_idx, z_loss)

@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr, labels_ptr, dloss_ptr, dlogits_ptr,
    num_classes, num_rows, smoothing, logit_scale, BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr, SPLIT: tl.constexpr
):
    # Define the row and column index
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)

    # Load the logits and lse for this row
    logits = tl.load(logits_ptr + row_idx * num_classes + col_idx, mask=col_idx < num_classes, other=-float('inf'))
    lse = tl.load(lse_ptr + row_idx)

    # Apply scaling to logits if necessary
    if logit_scale != 1.0:
        logits *= logit_scale

    # Compute probabilities
    max_logits = tl.max(logits, axis=0)
    logits_exp = tl.exp(logits - max_logits)
    probabilities = logits_exp / tl.sum(logits_exp, axis=0)

    # Get the label for this row
    label = tl.load(labels_ptr + row_idx)

    # Compute gradient
    dloss = tl.load(dloss_ptr + row_idx)
    grad = probabilities
    if label != -1:  # Assuming -1 is the ignored_index
        grad[label] -= 1.0
    if HAS_SMOOTHING:
        grad = (1.0 - smoothing) * grad + smoothing / num_classes

    # Store dlogits
    grad *= dloss
    tl.store(dlogits_ptr + row_idx * num_classes + col_idx, grad, mask=col_idx < num_classes)

# Wrapper functions
def cross_entropy_fwd(logits, labels, smoothing=0.0, logit_scale=1.0, BLOCK_SIZE=128, HAS_SMOOTHING=False, SPLIT=False):
    num_rows, num_classes = logits.shape
    loss = torch.empty(num_rows, device=logits.device, dtype=logits.dtype)
    lse = torch.empty(num_rows, device=logits.device, dtype=logits.dtype)
    z_loss = torch.empty(num_rows, device=logits.device, dtype=logits.dtype)

    grid = (num_rows,)
    cross_entropy_fwd_kernel[grid](
        logits, labels, loss, lse, z_loss, num_classes, num_rows, smoothing, logit_scale,
        BLOCK_SIZE=BLOCK_SIZE, HAS_SMOOTHING=HAS_SMOOTHING, SPLIT=SPLIT
    )

    return loss, lse, z_loss

def cross_entropy_bwd(logits, labels, dloss, smoothing=0.0, logit_scale=1.0, BLOCK_SIZE=128, HAS_SMOOTHING=False, SPLIT=False):
    num_rows, num_classes = logits.shape
    dlogits = torch.empty_like(logits)

    grid = (num_rows,)
    cross_entropy_bwd_kernel[grid](
        logits, labels, dloss, dlogits, num_classes, num_rows, smoothing, logit_scale,
        BLOCK_SIZE=BLOCK_SIZE, HAS_SMOOTHING=HAS_SMOOTHING, SPLIT=SPLIT
    )

    return dlogits
