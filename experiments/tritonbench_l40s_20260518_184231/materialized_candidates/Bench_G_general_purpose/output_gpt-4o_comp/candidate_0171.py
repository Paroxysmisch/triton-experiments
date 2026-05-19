import triton
import triton.language as tl
import torch

# Constants
MAX_FUSED_SIZE = 1024  # Maximum size for fused operations

def calculate_settings(n):
    """Determine the optimal block size and number of warps for GPU execution."""
    BLOCK_SIZE = min(MAX_FUSED_SIZE, n)
    NUM_WARPS = max(1, BLOCK_SIZE // 256)  # Number of warps based on block size
    return BLOCK_SIZE, NUM_WARPS

@triton.jit
def _cross_entropy_forward(
    logits_ptr, labels_ptr, loss_ptr,
    n_cols, logit_scale, softcap,
    BLOCK_SIZE: tl.constexpr
):
    """Compute cross-entropy loss for a single input row."""
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)

    # Load logits for this row
    logits = tl.load(logits_ptr + row_idx * n_cols + col_idx, mask=col_idx < n_cols, other=-float('inf'))

    # Apply optional logit scaling
    if logit_scale is not None:
        logits *= logit_scale

    # Compute max for numerical stability
    max_logits = tl.max(logits, axis=0)
    logits = logits - max_logits

    # Compute exp and sum for log-sum-exp
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    log_sum_exp = tl.log(sum_exp_logits)

    # Load label
    label = tl.load(labels_ptr + row_idx)

    # Compute loss if label is valid
    loss = 0.0
    if label != -100:  # Masked labels are ignored
        true_logit = tl.load(logits_ptr + row_idx * n_cols + label)
        loss = log_sum_exp - true_logit

    # Apply optional softcapping
    if softcap is not None:
        loss = tl.minimum(loss, softcap)

    # Store the loss
    tl.store(loss_ptr + row_idx, loss)

@triton.jit
def _chunked_cross_entropy_forward(
    logits_ptr, labels_ptr, loss_ptr,
    n_cols, logit_scale, softcap,
    BLOCK_SIZE: tl.constexpr
):
    """Compute cross-entropy loss for large vocab sizes in chunks."""
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)

    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    log_sum_exp = 0.0
    true_logit = 0.0

    for chunk_id in range(num_chunks):
        chunk_offset = chunk_id * BLOCK_SIZE
        chunk_mask = col_idx + chunk_offset < n_cols

        # Load logits for this chunk
        logits = tl.load(logits_ptr + row_idx * n_cols + chunk_offset + col_idx, mask=chunk_mask, other=-float('inf'))

        # Apply optional logit scaling
        if logit_scale is not None:
            logits *= logit_scale

        # Compute max for numerical stability
        max_logits = tl.max(logits, axis=0)
        logits = logits - max_logits

        # Compute exp and sum for log-sum-exp
        exp_logits = tl.exp(logits)
        log_sum_exp += tl.sum(exp_logits, axis=0)

        # Load true logit if label is in this chunk
        label = tl.load(labels_ptr + row_idx)
        if label >= chunk_offset and label < chunk_offset + BLOCK_SIZE:
            true_logit = tl.load(logits_ptr + row_idx * n_cols + label)

    # Finalize log-sum-exp calculation
    log_sum_exp = tl.log(log_sum_exp)

    # Compute loss if label is valid
    loss = 0.0
    if label != -100:  # Masked labels are ignored
        loss = log_sum_exp - true_logit

    # Apply optional softcapping
    if softcap is not None:
        loss = tl.minimum(loss, softcap)

    # Store the loss
    tl.store(loss_ptr + row_idx, loss)

@triton.jit
def _cross_entropy_backward(
    logits_ptr, labels_ptr, dlosses_ptr, dlogits_ptr,
    n_cols, logit_scale, softcap,
    BLOCK_SIZE: tl.constexpr
):
    """Compute gradients with respect to logits."""
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, BLOCK_SIZE)

    # Load logits for this row
    logits = tl.load(logits_ptr + row_idx * n_cols + col_idx, mask=col_idx < n_cols, other=-float('inf'))

    # Apply optional logit scaling
    if logit_scale is not None:
        logits *= logit_scale

    # Compute max for numerical stability
    max_logits = tl.max(logits, axis=0)
    logits = logits - max_logits

    # Compute exp and sum for softmax
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=0)
    softmax = exp_logits / sum_exp_logits

    # Load label
    label = tl.load(labels_ptr + row_idx)

    # Compute gradient
    grad = softmax
    if label != -100:  # Masked labels are ignored
        grad = grad - (col_idx == label)

    # Apply optional transformations
    if logit_scale is not None:
        grad *= logit_scale
    if softcap is not None:
        grad = tl.minimum(grad, softcap)

    # Scale by incoming gradient
    dloss = tl.load(dlosses_ptr + row_idx)
    grad *= dloss

    # Store the gradient
    tl.store(dlogits_ptr + row_idx * n_cols + col_idx, grad, mask=col_idx < n_cols)

class Fast_CrossEntropyLoss:
    def __init__(self, logit_scale=None, softcap=None):
        self.logit_scale = logit_scale
        self.softcap = softcap

    def forward(self, logits, labels):
        B, N = logits.shape
        BLOCK_SIZE, NUM_WARPS = calculate_settings(N)

        loss = torch.empty(B, device=logits.device, dtype=logits.dtype)
        if N <= MAX_FUSED_SIZE:
            _cross_entropy_forward[(B,)](
                logits, labels, loss,
                N, self.logit_scale, self.softcap,
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            _chunked_cross_entropy_forward[(B,)](
                logits, labels, loss,
                N, self.logit_scale, self.softcap,
                BLOCK_SIZE=BLOCK_SIZE
            )
        return loss.mean()

    def backward(self, logits, labels, dlosses):
        B, N = logits.shape
        BLOCK_SIZE, NUM_WARPS = calculate_settings(N)

        dlogits = torch.empty_like(logits)
        _cross_entropy_backward[(B,)](
            logits, labels, dlosses, dlogits,
            N, self.logit_scale, self.softcap,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return dlogits

def fast_cross_entropy_loss(logits, labels, logit_scale=None, softcap=None):
    """Compute the mean cross-entropy loss."""
    loss_fn = Fast_CrossEntropyLoss(logit_scale=logit_scale, softcap=softcap)
    return loss_fn.forward(logits, labels)
