import torch
import triton
import triton.language as tl

@triton.jit
def _fused_xent_sfmax_ln_kernel(
    logits_ptr,            # [n, c]
    targets_ptr,           # [n] or [n, c]
    out_ptr,               # [n, c] if out is not None, else may be None
    loss_ptr,              # [n], partial cross-entropy loss per row
    weight_ptr,            # [c] or None
    n_elements,            # total elements in logits (n*c)
    n,                     # batch size
    c,                     # number of classes
    ignore_index,
    label_smoothing,
    eps,
    is_indices,            # bool
    use_weight,            # bool
    BLOCK_SIZE: tl.constexpr
):
    # row index
    row_idx = tl.program_id(0)
    # each program handles exactly one row
    # return if out of range
    if row_idx >= n:
        return

    # --------------------------------------------------
    # 1) Load logits for this row, compute max for numerical stability
    # --------------------------------------------------
    offs = row_idx * c + tl.arange(0, BLOCK_SIZE)
    mask = offs < (row_idx * c + c)
    # read logits
    logits = tl.where(mask, tl.load(logits_ptr + offs, mask=mask), float("-inf"))
    # row-wise max
    max_logit = tl.maximum(tl.max(logits, axis=0), 0.)  # safe to reduce with numeric stabilities
    # broadcast
    logits -= max_logit

    # --------------------------------------------------
    # 2) Compute exponentials & sum for softmax
    # --------------------------------------------------
    exps = tl.exp(logits)
    denom = tl.sum(exps, axis=0) + 1e-9
    probs = exps / denom

    # --------------------------------------------------
    # 3) Cross-Entropy loss
    # --------------------------------------------------
    # We accumulate cross-entropy in local var. Will store to loss_ptr[row_idx].
    ce_loss = 0.0

    if is_indices:
        # read target index for this row
        tval = tl.load(targets_ptr + row_idx)
        # ignore if tval == ignore_index
        if tval != ignore_index and tval >= 0 and tval < c:
            # label smoothing
            smooth_prob = label_smoothing / float(c)
            p_target = tl.load(probs, mask=(tl.arange(0, BLOCK_SIZE) == tval))  # probability at target
            p_target = tl.sum(p_target, axis=0)  # reduce to scalar
            # cross entropy = -log((1 - label_smoothing)*p_target + smooth_prob)
            # clamp to avoid log(0)
            cross_p = (1.0 - label_smoothing) * p_target + smooth_prob
            cross_p = tl.maximum(cross_p, 1e-9)
            val = -tl.log(cross_p)
            if use_weight:
                w_val = tl.load(weight_ptr + tval)
                val *= w_val
            ce_loss = val
        else:
            ce_loss = 0.0
    else:
        # target is distribution. shape [n, c]
        # load distribution row slice
        tgt = tl.where(mask, tl.load(targets_ptr + offs, mask=mask), 0.0)
        # apply label smoothing:
        # y_ij = y_ij*(1 - label_smoothing) + label_smoothing/C
        smoothed = tgt * (1.0 - label_smoothing) + label_smoothing / float(c)
        # cross-entropy = - sum_j [ smoothed_j * log(prob_j) ]
        # clamp prob
        prob_clamped = tl.maximum(probs, 1e-9)
        val = -smoothed * tl.log(prob_clamped)
        if use_weight:
            w = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask)
            val = val * w
        ce_loss = tl.sum(val, axis=0)

    # store partial CE in loss_ptr
    tl.store(loss_ptr + row_idx, ce_loss)

    # --------------------------------------------------
    # 4) LayerNorm on probs -> out
    # --------------------------------------------------
    # LN over dimension c
    mean_p = tl.sum(probs, axis=0) / float(c)
    var_p = tl.sum((probs - mean_p)*(probs - mean_p), axis=0) / float(c)
    var_p = var_p + eps
    rstd = 1.0 / tl.sqrt(var_p)
    norm_p = (probs - mean_p) * rstd

    # store LN result if out_ptr is not None
    if out_ptr != 0:
        tl.store(out_ptr + offs, tl.where(mask, norm_p, 0.0), mask=mask)


def fused_cross_entropy_softmax_layernorm(
    logits: torch.Tensor,
    targets: torch.Tensor,
    normalized_shape,
    weight=None,
    ignore_index: int = -100,
    reduction: str = 'mean',
    label_smoothing: float = 0.0,
    eps: float = 1e-5,
    *,
    out=None
) -> (torch.Tensor, torch.Tensor):
    """
    Fused operation: cross-entropy loss, softmax, and layer normalization.
    Returns a tuple of (loss, normalized_output).
    """
    # Ensure 2D shape for kernel example: (N, C)
    # If more dimensions, reshape for demonstration.
    logits_2d = logits.view(logits.size(0), -1)
    n, c = logits_2d.shape
    # Decide if targets are indices or distribution
    is_indices = False
    if targets.dim() == 1 or (targets.dim() == 2 and targets.shape[1] == 1):
