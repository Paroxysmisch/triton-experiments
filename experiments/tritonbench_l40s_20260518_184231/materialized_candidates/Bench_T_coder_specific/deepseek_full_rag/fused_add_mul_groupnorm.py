= input.to(tl.float32)

    count = prev_count + curr_count
    delta = input - prev_mean
    mean = prev_mean + delta * curr_count / count
    g_var = prev_var + tl.sum(delta * (input - mean) * mask, axis=0)

    return count, mean, g_var

@triton.jit
def update_ema(prev_ema, new_val, count, momentum):
    """
    Updates exponential moving average.

    Args:
        prev_ema: Previous exponential moving average.
        new_val: New value used to update the exponential moving average.
            The new value must be of the same shape as the momentum.
        count: Count of updates since the last exponential moving average update.
        momentum: Momentum to apply when updating the exponential moving average.
            The momentum must be of the same shape as the new value.

    Returns:
        Updated exponential moving average.
    """
    return (1 - momentum) * prev_ema + momentum * new_val / count

@triton.jit
def standardize(input, mean, inv_std, eps, mask: tl.constexpr):
    """
    Standardizes the input using mean and inverse standard deviation.

    Args:
        input: Input to standardize.
            The input must be of the same shape as the mask.
        mean: Mean to subtract from the input.
            The mean must be of the same shape as the input.
        inv_std: Inverse standard deviation to multiply the input by.
            The inverse standard deviation must be of the same shape as the input.
        eps: Epsilon added in the numerator to avoid division by zero.
            The epsilon must be a scalar.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of the same shape as the input.

    Returns:
        Standardized input.
    """
    input = input.to(tl.float32)

    return (input - mean[:, None]) * inv_std[:, None] * tl.rsqrt(eps + tl.sum((input - mean[:, None]) * (input - mean[:, None]) * mask, axis=0))

@triton.jit
def calc_l1l2_norm(input, order, mask: tl.constexpr):
    """
    Calculates the L1 or L2 norm of the input.

    Args:
        input: Input whose L1 or L2 norm is calculated.
            The input must be of the same shape as the mask.
        order: Order of the norm to calculate.
            The order must be a scalar.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of the same shape as the input.

    Returns:
        L1 or L2 norm of the input.
    """
    input = input.to(tl.float32)

    return tl.sum(tl.abs(input) ** order * mask, axis=0) ** (1 / order)

@triton.jit
def calc_neg_log_likelihood(input, target, mask: tl.constexpr):
    """
    Calculates the negative log likelihood loss between the input and target.

    Args:
        input: Input tensor representing probabilities.
            The input must be of the same shape as the mask and must be between 0 and 1.
        target: Target tensor representing ground truth labels.
            The target must be of the same shape as the mask and must be binary.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of the same shape as the input and target.

    Returns:
        Negative log likelihood loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    input = tl.where(mask, input, 0)
    target = tl.where(mask, target, 0)

    input = tl.clip(input, 1e-12, 1 - 1e-12)
    output = -tl.sum(target * tl.log(input) + (1 - target) * tl.log(1 - input))

    return output

@triton.jit
def calc_cross_entropy(input, target, ignore_index, mask: tl.constexpr):
    """
    Calculates the cross entropy loss between the input and target.

    Args:
        input: Input tensor representing probabilities.
            The input must be of the same shape as the mask and must be between 0 and 1.
        target: Target tensor representing ground truth labels.
            The target must be of the same shape as the mask.
        ignore_index: Index to ignore in the calculation of the loss.
            The ignore index must be a scalar.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of the same shape as the input and target.

    Returns:
        Cross entropy loss between the input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)

    input = tl.where(mask, input, 0)
    target = tl.where(mask, target, ignore_index)

    input = tl.clip(input, 1e-12, 1 - 1e-12)
    output = -tl.sum(tl.where(target != ignore_index, target * tl.log(input), 0)) / tl.sum(target != ignore_index)

    return output
