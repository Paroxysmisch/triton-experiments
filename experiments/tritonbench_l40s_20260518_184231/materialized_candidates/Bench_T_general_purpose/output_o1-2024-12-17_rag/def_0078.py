import triton
import triton.language as tl
import torch

@triton.jit
def _fused_cross_entropy_log_softmax_kernel(
    logits_ptr,            # [n, c], row-major
    targets_ptr,           # [n], row-major
    weights_ptr,           # [c] or None
    output_ptr,            # [n], row-major
    valid_ptr,             # [n], row-major (0 or 1, for ignored vs valid)
    n_elements,            # total number of elements = n * c
    num_classes,           # c
    label_smoothing,
    ignore_index,
    has_weight: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)  # one block per row
    row_start = pid * num_classes
    # If out of range, just return
    if row_start >= n_elements:
        return

    # Load target index
    t = tl.load(targets_ptr + pid)
    # Mark valid row or not
    is_valid = 1
    if t == ignore_index:
        is_valid = 0

    # ----------------------------------------------
    # Pass 1: find row-wise max for numerical stability
    max_val = float('-inf')
    # We iterate over this row in chunks
    for i in range(0, num_classes, BLOCK_SIZE):
        idx = row_start + i + tl.arange(0, BLOCK_SIZE)
        mask = idx < (row_start + num_classes)
        val = tl.where(mask, tl.load(logits_ptr + idx), float('-inf'))
        block_max = tl.maximum(val, -float('inf'))
        curr_max = tl.max(block_max, axis=0)
        max_val = tl.maximum(max_val, curr_max)
    # ----------------------------------------------
    # Pass 2: compute log-sum-exp and partial sums for label smoothing
    sum_exp = 0.0
    row_sum_logprob = 0.0     # sum of log_probs for label smoothing
    target_logprob = 0.0      # log_prob for the correct class
    for i in range(0, num_classes, BLOCK_SIZE):
        idx = row_start + i + tl.arange(0, BLOCK_SIZE)
        mask = idx < (row_start + num_classes)
        val = tl.where(mask, tl.load(logits_ptr + idx), float('-inf'))
        val_f = val - max_val  # shift for numerical stability
        exp_val = tl.exp(val_f)
        sum_block = tl.sum(exp_val, axis=0)
        sum_exp += sum_block

    # Now have sum_exp, so we can compute log_probs in a second pass
    denom = tl.log(sum_exp)   # log of denominator
    for i in range(0, num_classes, BLOCK_SIZE):
        idx = row_start + i + tl.arange(0, BLOCK_SIZE)
        mask = idx < (row_start + num_classes)
        val = tl.where(mask, tl.load(logits_ptr + idx), float('-inf'))
        log_p = val - max_val - denom
        # Accumulate sum of log_probs for label smoothing
        block_sum = tl.where(mask, log_p, 0.0)
        row_sum_logprob += tl.sum(block_sum, axis=0)
        # Gather target logprob
        if is_valid == 1:
            # check if this block covers the target index
            # t is the class index, so we compare idx
            target_mask = (tl.arange(0, BLOCK_SIZE) + i) == t
            masked_log_p = tl.where(target_mask & mask, log_p, 0.0)
            target_logprob += tl.sum(masked_log_p, axis=0)

    # ----------------------------------------------
    # Compute cross entropy per row
    # If is_valid == 0, we will store 0 in output
    # If label_smoothing > 0:
    #   CE = - [ (1 - ls) * log_p_target + (ls / c) * sum_{c} log_p_c ]
    # else:
    #   CE = - log_p_target
    ce_val = 0.0
    if is_valid == 1:
        if label_smoothing > 0.0:
            one_minus_ls = 1.0 - label_smoothing
            ls_div_c = label_smoothing / float(num_classes)
            ce_val = - (one_minus_ls * target_logprob + ls_div_c * row_sum_logprob)
        else:
            ce_val = - target_logprob

        # if weight is provided, multiply
        if has_weight:
            w = tl.load(weights_ptr + t)
            ce_val = ce_val * w

    # store result
    tl.store(output_ptr + pid, ce_val)
    tl.store(valid_ptr + pid, float(is_valid))


def fused_cross_entropy_log_softmax(
    input: torch.Tensor,
    target: torch.Tensor,
    dim: int = 1,
    weight: torch.Tensor = None,
    ignore_index: int = -100,
    reduction: str = 'mean',
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Fused cross entropy with log softmax for improved numerical stability.

    Args:
        input (Tensor): Input tensor of logits, where softmax/log_softmax
            will be computed along `dim`.
        target (Tensor): Ground truth class indices or probabilities.
        dim (int, optional): Dimension along which to compute log softmax. Default is 1.
        weight (Tensor, optional): Manual rescaling weight for each class.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default: -100.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.
        label_smoothing (float, optional): Specifies the amount of smoothing to be applied,
            where 0.0 means no smoothing. Default: 0.0.

    Returns:
        Tensor: The computed cross entropy loss. The shape of the output depends on `reduction`.
    """
    # Ensure dim is within range
    if dim < 0:
        dim = input.dim() + dim
    # Move `dim` to the last dimension if not already
    if dim != input.dim() - 1:
        # Permute so that `dim` becomes the last dimension
        dims
