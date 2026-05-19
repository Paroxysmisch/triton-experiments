import math
import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr,            # [num_rows, num_cols]
    labels_ptr,            # [num_rows]
    loss_ptr,              # [num_rows]
    lse_ptr,               # [num_rows]
    dist_class_start_ptr,  # scalar
    smoothing_ptr,         # scalar
    ignore_index_ptr,      # scalar
    scale_ptr,             # scalar
    n_col,                 # logits.shape[1]
    BLOCK_SIZE: tl.constexpr
):
    # Program index for row
    row_id = tl.program_id(0)
    # Pointers
    dist_class_start = tl.load(dist_class_start_ptr)
    smoothing = tl.load(smoothing_ptr)
    ignore_index = tl.load(ignore_index_ptr)
    scale = tl.load(scale_ptr)
    # Check if in range
    if row_id * BLOCK_SIZE >= n_col:
        return

    # Read label
    label = tl.load(labels_ptr + row_id)
    # offset for the row in logits and loss
    row_start = row_id * n_col

    # We loop along columns in BLOCK_SIZE chunks
    idxs = row_start + (tl.arange(0, BLOCK_SIZE) % BLOCK_SIZE)
    # Initialize partial max
    max_val = tl.full([BLOCK_SIZE], float('-inf'), tl.float32)
    # We iterate across columns to find max for log-sum-exp
    for start_col in range(0, n_col, BLOCK_SIZE):
        idx = row_start + start_col + (tl.arange(0, BLOCK_SIZE) % BLOCK_SIZE)
        mask = idx < (row_start + n_col)
        val = tl.where(
            mask,
            tl.load(logits_ptr + idx) * scale,
            float('-inf')
        )
        max_val = tl.maximum(max_val, val)

    # Compute exp form for partial sums
    sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start_col in range(0, n_col, BLOCK_SIZE):
        idx = row_start + start_col + (tl.arange(0, BLOCK_SIZE) % BLOCK_SIZE)
        mask = idx < (row_start + n_col)
        val = tl.where(
            mask,
            tl.load(logits_ptr + idx) * scale,
            float('-inf')
        )
        val = tl.exp(val - max_val)
        sum_exp += tl.where(mask, val, 0.)

    # Log-sum-exp
    lse_val = tl.log(sum_exp) + max_val[0]  # all threads have the same max_val across the block dim
    tl.store(lse_ptr + row_id, lse_val)

    # Compute cross entropy loss
    # for row_id, check if label is in this distributed range if needed
    row_loss = 0.
    # If ignoring this index
    if label == ignore_index:
        row_loss = 0.
    else:
        # shift label if dist_class_start > 0
        correct_label = label - dist_class_start
        # check if label is out of range
        is_valid_label = (correct_label >= 0) & (correct_label < n_col)
        # label smoothing
        if is_valid_label:
            correct_label_val = tl.load(logits_ptr + (row_start + correct_label.to(tl.int32))) * scale
            row_loss = (lse_val - correct_label_val) * (1. - smoothing)
            if smoothing > 0.:
                row_loss += smoothing * lse_val
        else:
            if smoothing > 0.:
                row_loss = smoothing * lse_val
            else:
                # invalid label => no contribution
                row_loss = 0.

    tl.store(loss_ptr + row_id, row_loss)


@triton.jit
def cross_entropy_bwd_kernel(
    logits_ptr,            # [num_rows, num_cols]
    labels_ptr,            # [num_rows]
    dlogits_ptr,           # [num_rows, num_cols]
    lse_ptr,               # [num_rows]
    dist_class_start_ptr,  # scalar
    smoothing_ptr,         # scalar
    ignore_index_ptr,      # scalar
    scale_ptr,             # scalar
    grad_out_ptr,          # scalar (d_loss w.r.t. out)
    n_row,                 # logits.shape[0]
    n_col,                 # logits.shape[1]
    BLOCK_SIZE: tl.constexpr
):
    # Program index for columns
    col_id = tl.program_id(0)
    # Pointers
    dist_class_start = tl.load(dist_class_start_ptr)
    smoothing = tl.load(smoothing_ptr)
    ignore_index = tl.load(ignore_index_ptr)
    scale = tl.load(scale_ptr)
    grad_out = tl.load(grad_out_ptr)

    # Range for the block in columns
    offsets = col_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Loop over the rows
    for row_id in range(n_row):
        if row_id >= n_row:
            return
        row_start = row_id * n_col
        # No contribution if label was ignored
        label = tl.load(labels_ptr + row_id)
        if label == ignore_index:
            mask = offsets < n_col
            idx = row_start + offsets
            tl.store(dlogits_ptr + idx, 0., mask=mask)
            continue

        # shift label for potential distribution offset
        correct_label = label - dist_class_start
        is_valid_label = (correct_label >= 0) & (correct_label < n_col)
        # load lse
        lse_val = tl.load(lse_ptr + row_id)
        # mask for in-range columns
        mask = offsets < n_col
        idx = row_start + offsets
        # scales
        logit_val = tl.load(logits_ptr + idx, mask=mask) * scale
        logp = logit_val - lse_val
        exp_val = tl.exp(logp)
        prob = exp_val

        # Adjust for smoothing
        if smoothing > 0.:
            prob = (1. - smoothing) * prob + (smoothing * 1.0 / float(n_col))

        grad = prob  # default gradient from softmax
        if is_valid_label:
            # subtract 1.0 for correct class if in range (1-smoothing factor)
            correct_mask = (offsets == correct_label.to(tl.int32))
            if smoothing > 0.:
                grad -= (1. - smoothing) * correct_mask
            else:
                grad -= correct_mask

        # scale dL/dlogits
        grad_out_scaled = grad_out * grad
        # Because we multiplied logits by scale, chain rule => multiply by scale factor
        grad_out_scaled *= scale

        tl.store(dlogits_ptr + idx, grad_out_scaled, mask=mask)


class CrossEntropyLoss:
    """
    Triton-based Cross Entropy with optional label smoothing and distribution offset.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        smoothing: float = 0.0,
        scale: float = 1.0,
        dist_class_start: int = 0
    ):
        self.ignore_index = ignore_index
        self.smoothing = smoothing
        self.scale = scale
        self.dist_class_start = dist_class_start

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        logits: [batch_size, vocab_size]
        labels: [batch_size]
        """
        assert logits.is_cuda, "Logits must be on CUDA"
        batch_size, vocab_size = logits.shape
        loss = torch.empty(batch_size, dtype=logits.dtype, device=logits.device)
        lse = torch.empty(batch_size, dtype=logits.dtype, device=logits.device)

        grid = (batch_size,)
        cross_entropy_fwd_kernel[grid](
            logits_ptr=logits.data_ptr(),
            labels_ptr=labels.data_ptr(),
            loss_ptr=loss.data_ptr(),
            lse_ptr=lse.data_ptr(),
            dist_class_start_ptr=torch.tensor([self.dist_class_start], dtype=logits.dtype, device=logits.device).data_ptr(),
            smoothing_ptr=torch.tensor([self.smoothing], dtype=logits.dtype, device=logits.device).data_ptr(),
            ignore_index_ptr=torch.tensor([self.ignore_index], dtype=labels.dtype, device=labels.device).data_ptr(),
            scale_ptr=torch.tensor([self.scale], dtype=logits.dtype, device=logits.device).data_ptr(),
            n_col=vocab_size,
            BLOCK_SIZE=128
        )
        return loss, lse

    def backward(self, logits: torch.Tensor, labels: torch.Tensor, lse: torch.Tensor, grad_out: torch.Tensor, out=None):
        """
        logits: [batch_size, vocab_size]
        labels: [batch_size]
        lse:    [batch_size]
        grad_out: scalar or [1] tensor representing gradient from subsequent layer
        """
        assert logits.is_cuda, "Logits must be on CUDA"
        batch_size, vocab_size = logits.shape
        if out is None:
            dlogits = torch.empty_like(logits)
        else:
            dlogits = out

        grid = (math.ceil(vocab_size / 128),)
        cross_entropy_bwd_kernel[grid](
            logits_ptr=logits.data_ptr(),
            labels_ptr=labels.data_ptr(),
            dlogits_ptr=dlogits.data_ptr(),
            lse_ptr=lse.data_ptr(),
            dist_class_start_ptr=torch.tensor([self.dist_class_start], dtype=logits.dtype, device=logits.device).data_ptr(),
            smoothing_ptr=torch.tensor([self.smoothing], dtype=logits.dtype, device=logits.device).data_ptr(),
            ignore_index_ptr=torch.tensor([self.ignore_index], dtype=labels.dtype, device=labels.device).data_ptr(),
            scale_ptr=torch.tensor([self.scale], dtype=logits.dtype, device=logits.device).data_ptr(),
            grad_out_ptr=grad_out.data_ptr(),
            n_row=batch_size,
            n_col=vocab_size,
            BLOCK_SIZE=128
        )
        return dlogits


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    scale: float = 1.0,
    dist_class_start: int = 0
):
    """
    High-level function for cross entropy with Triton support
    """
    celoss = CrossEntropyLoss(ignore_index, label_smoothing, scale, dist_class_start)
    loss, lse = celoss.forward(logits, labels)
    return loss, lse
