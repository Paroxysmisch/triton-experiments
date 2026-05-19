tl.constexpr):
    """
    Updates a Welford's algorithm statistics with the input.

    Args:
        input: Input whose statistics are updated.
            The input must be of shape [BLOCK_SIZE].
        prev_count: Count from the previous statistics.
        prev_mean: Mean from the previous statistics.
        prev_var: Variance from the previous statistics.
        curr_count: Count from the current input.
        mask: Mask indicating which elements should be included in the update.
            The mask must be of shape [BLOCK_SIZE].

    Returns:
        Updated count, mean, and variance.
    """
    input = input.to(tl.float32)

    curr_mean = tl.sum(input) / curr_count
    diff = tl.where(mask, input - curr_mean, 0)
    curr_var = tl.sum(diff * diff) / curr_count

    count = prev_count + curr_count
    mean = (prev_mean * (prev_count / count) + curr_mean * (curr_count / count))
    var = (prev_var * (prev_count / count) + curr_var * (curr_count / count))

    return count, mean, var

@triton.jit
def update_ema(input, prev_ema, decay, last_dim_mask: tl.constexpr):
    """
    Updates an exponential moving average with the input.

    Args:
        input: Input whose exponential moving average is updated.
            The input must be of shape [BLOCK_SIZE].
        prev_ema: Exponential moving average from the previous input.
        decay: Decay factor.
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the update.
            The mask must be of shape [BLOCK_SIZE].

    Returns:
        Updated exponential moving average.
    """
    input = input.to(tl.float32)

    return tl.where(last_dim_mask, input * decay + prev_ema * (1 - decay), prev_ema)

@triton.jit
def normalize_input(input, mean, inv_std, last_dim_mask: tl.constexpr):
    """
    Normalizes the input using mean and inverse standard deviation.

    Args:
        input: Input to normalize.
            The input must be of shape [BLOCK_SIZE].
        mean: Mean to subtract from the input.
        inv_std: Inverse standard deviation to multiply the input by.
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the normalization.
            The mask must be of shape [BLOCK_SIZE].

    Returns:
        Normalized input.
    """
    input = input.to(tl.float32)

    return tl.where(last_dim_mask, (input - mean) * inv_std, 0)

@triton.jit
def calc_l1l2_norm_loss(input, target, reduction_type, last_dim_mask: tl.constexpr):
    """
    Calculates the L1/L2 norm loss of the input with respect to the target.

    Args:
        input: Input whose norm loss is calculated.
            The input must be of shape [BLOCK_SIZE].
        target: Target whose norm is used as the denominator.
            The target must be of shape [BLOCK_SIZE].
        reduction_type: Type of reduction to perform.
            Options are 'none', 'mean', 'sum', and 'batchwise_mean'.
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the loss calculation.
            The mask must be of shape [BLOCK_SIZE].

    Returns:
        L1/L2 norm loss of the input with respect to the target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    diff = tl.where(last_dim_mask, input - target, 0)
    abs_diff = tl.abs(diff)
    output = tl.where(last_dim_mask, abs_diff / target, 0)

    if reduction_type == 'none':
        return output

    elif reduction_type == 'mean':
        count = tl.sum(last_dim_mask)
        return tl.sum(output) / count

    elif reduction_type == 'sum':
        return tl.sum(output)

    elif reduction_type == 'batchwise_mean':
        batch_count = tl.sum(last_dim_mask, axis=0)
        return tl.sum(output, axis=0) / batch_count

@triton.jit
def calc_neg_log_likelihood_loss(input, target, last_dim_mask: tl.constexpr):
    """
    Calculates the negative log likelihood loss of the input with respect to the target.

    Args:
        input: Input whose neg log likelihood loss is calculated.
            The input must be of shape [BLOCK_SIZE].
        target: Target whose value is used as the index of the max value
            in the input for each batch. The target must be of shape [BLOCK_SIZE].
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the loss calculation.
            The mask must be of shape [BLOCK_SIZE].

    Returns:
        Negative log likelihood loss of the input with respect to the target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.int64)

    logits = tl.maximum(input, 0)
    logits = logits - tl.log(tl.sum(tl.exp(logits), axis=0))

    output = -tl.where(last_dim_mask, logits[target], 0)

    return output

@triton.jit
def calc_cross_entropy_loss(input, target, last_dim_mask: tl.constexpr):
    """
    Calculates the cross entropy loss of the input with respect to the target.

    Args:
        input: Input whose cross entropy loss is calculated.
            The input must be of shape [BLOCK_SIZE].
        target: Target whose value is used as the index of the max value
            in the input for each batch. The target must be of shape [BLOCK_SIZE].
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the loss calculation.
            The mask must be of shape [BLOCK_SIZE].

    Returns:
        Cross entropy loss of the input with respect to the target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.int64)

    logits = tl.log(tl.maximum(input, 1e-14))

    output = tl.where(last_dim_mask, logits[target], 0)

    return -output
