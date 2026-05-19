import torch
import triton
import triton.language as tl

@triton.jit
def _fused_ce_sm_ln_kernel(
    logits_ptr, targets_ptr, output_ptr, loss_ptr,
    N, C, eps, ignore_idx,
    label_smoothing, use_target_probs, apply_weight,
    weight_ptr, apply_ignore,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    row_end = tl.minimum(row_start + BLOCK_SIZE, N)
    row_offsets = tl.arange(0, BLOCK_SIZE)
    mask_row = row_offsets < (row_end - row_start)

    # -----------------------------
    # Load logits and optionally targets
    # -----------------------------
    # Each row of shape [C]
    # We'll load [BLOCK_SIZE x C] chunk
    # row = row_start + row_offsets
    logits = tl.zeros([BLOCK_SIZE, C], dtype=tl.float32)
    targets = tl.zeros([BLOCK_SIZE, C], dtype=tl.float32)

    for c in range(C):
        offset_logit = (row_start + row_offsets) * C + c
        mask = mask_row & (offset_logit < N*C)
        logits_val = tl.load(logits_ptr + offset_logit, mask=mask, other=0.)
        logits_val = logits_val.to(tl.float32)
        logits = tl.where(mask[:, None], logits, logits)
        logits[:, c] = tl.where(mask, logits_val, logits[:, c])

    # For cross-entropy
    if use_target_probs:
        for c in range(C):
            offset_target = (row_start + row_offsets) * C + c
            mask = mask_row & (offset_target < N*C)
            target_val = tl.load(targets_ptr + offset_target, mask=mask, other=0.)
            target_val = target_val.to(tl.float32)
            targets[:, c] = tl.where(mask, target_val, targets[:, c])
    else:
        # load target indices
        offset_t = row_start + row_offsets
        mask_t = mask_row & (offset_t < N)
        t_val = tl.where(
            mask_t,
            tl.load(targets_ptr + offset_t, mask=mask_t, other=0.),
            -1
        )
        # ignore_idx
        if apply_ignore:
            t_val = tl.where(t_val == ignore_idx, -1, t_val)

    # -----------------------------
    # Softmax
    # -----------------------------
    max_val = tl.max(logits, 1)
    logits = logits - max_val[:, None]
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, 1)
    softmax_vals = exp_logits / sum_exp[:, None]

    # -----------------------------
    # Cross Entropy
    # -----------------------------
    # Per-row partial loss
    loss_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    if use_target_probs:
        # label smoothing: p_target = (1 - label_smoothing) * targets + label_smoothing / C
        if label_smoothing > 0.0:
            smoothed = label_smoothing / float(C)
            for c in range(C):
                p_target = (1.0 - label_smoothing) * targets[:, c] + smoothed
                term = -p_target * (logits[:, c] - tl.log(sum_exp))
                loss_acc += term
        else:
            # normal cross-entropy
            for c in range(C):
                t_probs = targets[:, c]
                term = -t_probs * (logits[:, c] - tl.log(sum_exp))
                loss_acc += term
        # weight
        if apply_weight:
            # for probabilities: weight each column by class weight
            # approximate weighting by summation of target distribution
            # sum of t_probs for each class * weight[class]
            wsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            for c in range(C):
                t_probs = targets[:, c]
                wv = tl.load(weight_ptr + c)
                wsum += t_probs * wv
            loss_acc *= wsum
    else:
        # class indices cross entropy
        # label_smoothing if needed
        if label_smoothing > 0.0:
            smoothed = label_smoothing / float(C)
            # negative log-likelihood with smoothing
            # target prob for correct class: (1 - label_smoothing) + smoothed
            # for other classes: smoothed
            for c in range(C):
                correct_mask = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
                correct_mask = correct_mask + tl.where((c == t_val) & (t_val >= 0), 1.0, 0.0)
                p_target = correct_mask * (1.0 - label_smoothing) + smoothed
                term = -p_target * (logits[:, c] - tl.log(sum_exp))
                if apply_weight:
                    wv = tl.load(weight_ptr + c)
                    term *= wv
                loss_acc += term
        else:
            for c in range(C):
                # ignore rows where t_val < 0
                is_correct = (c == t_val) & (t_val >= 0)
                # negative log-likelihood
                sub_log = logits[:, c] - tl.log(sum_exp)
                term = -tl.where(is_correct, sub_log, 0.)
                if apply_weight:
                    wv = tl.load(weight_ptr + c)
                    term = term * tl.where(is_correct, wv, 1.0)
                loss_acc += term

    # -----------------------------
    # Layer Norm on probability
    # -----------------------------
    # mean of softmax_vals
    mean = tl.sum(softmax_vals, 1) / C
    var_part = softmax_vals - mean[:, None]
    var = tl.sum(var_part * var_part, 1) / C
    inv_std = 1.0 / tl.sqrt(var + eps)
    ln_out = (softmax_vals - mean[:, None]) * inv_std[:, None]

    # store output
    for c in range(C):
        offset_out = (row_start + row_offsets) * C + c
        mask = mask_row & (offset_out < N*C)
        val = ln_out[:, c]
        tl.store(output_ptr + offset_out, val, mask=mask)

    # store partial loss
    for i in range(BLOCK_SIZE):
        if mask_row[i]:
            offset_loss = (row_start + i)
            if offset_loss < N:
                tl.store(loss_ptr + offset_loss, loss_acc[i])

@triton.jit
def _reduce_loss_kernel(
    loss_ptr, out_ptr, N,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    val = tl.load(loss_ptr + offsets, mask=mask, other=0.)
    s = tl.sum(val, axis=0)
    if reduction == 0:  # sum
        tl.atomic_add(out_ptr, s)
    else:
        # for mean, we also do sum. We'll divide later by N in Python.
        tl.atomic_add(out_ptr, s)

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
    out: torch.Tensor = None
):
    """
    Fused cross entropy, softmax, and layer norm operation.
    See the function doc for parameter details.
    """
    assert logits.is_cuda, "logits must be a CUDA tensor"
    assert targets.is_cuda, "targets must be a CUDA tensor"
    if out is not None:
        assert out.is_cuda, "out must be a CUDA tensor"

    # Flatten any extra dimensions for computational simplicity, except keep (N, C)
    # If shape is (N, C, *), flatten to 2D with N' = N*prod(*)
    N = logits.shape[0]
    if len(logits.shape) >= 2:
        C = logits.shape[1]
        extra_shape = logits.shape[2:]
        N_ext = 1
        for s in extra_shape:
            N_ext *= s
        N = N * N_ext
    logits_2d = logits.contiguous().view(N, C)

    if targets.dim() == logits.dim():
        # target probabilities
        use_target_probs = True
    else:
        # target indices
        use_target_probs = False
    
    # Create storage for partial cross entropy
    loss_buffer = torch.empty(N, dtype=logits.dtype, device=logits.device)
    # Create output if needed
    if out is None:
        out = torch.empty_like(logits_2d)

    # Prepare weight if given
    apply_weight = (weight is not None)
    w_ptr = weight
    if apply_weight:
        w_ptr = weight.to(logits.device).to(logits.dtype)

    apply_ignore = (ignore_index is not None and ignore_index >= 0)

    BLOCK_SIZE = 1024
    grid = ( (N + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    _fused_ce_sm_ln_kernel[grid](
        logits_2d, targets, out, loss_buffer,
        N, C, eps, ignore_index,
        label_smoothing, use_target_probs, apply_weight,
        w_ptr if apply_weight else torch.empty(0, device=logits.device),
        apply_ignore,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Now reduce the partial losses based on reduction
    final_loss = torch.empty(1, dtype=logits.dtype, device=logits.device)
    final_loss[0] = 0.
    if reduction == 'none':
        # Just return the loss per-element and no reduction
        # shape: (N,) -> reshape to match original if needed
        # We'll keep the partial in loss_buffer
        loss_out = loss_buffer.view(logits.shape[0], *logits.shape[2:])
        return (loss_out, out.view(logits.shape))
    elif reduction in ('sum', 'mean'):
        red_code = 0 if reduction == 'sum' else 1
        _reduce_loss_kernel[1](
            loss_buffer, final_loss, N,
            reduction=red_code,
            BLOCK_SIZE=BLOCK_SIZE
        )
        if reduction == 'mean':
            final_loss = final_loss / float(N)
        return (final_loss, out.view(logits.shape))
    else:
        raise ValueError("Invalid reduction type")

    # Return
    return (final_loss, out.view(logits.shape))
