input.to(tl.float32)

    delta = tl.where(mask, input - prev_mean, 0)
    new_mean = prev_mean + delta * curr_count / (prev_count + curr_count)
    new_var = prev_var + tl.sum(delta * (input - new_mean) * (input - prev_mean), axis=1) / (prev_count + curr_count)

    return prev_count + curr_count, new_mean, new_var

@triton.jit
def update_ema(input, prev_ema, decay, mask: tl.constexpr):
    """
    Updates exponential moving average (EMA) for input.

    Args:
        input: Input used to update EMA.
            The input must be of the same shape as the mask.
        prev_ema: Previous EMA statistic to update.
        decay: Decay factor for EMA update.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of the same shape as the input.

    Returns:
        Updated EMA statistic.
    """
    input = input.to(tl.float32)

    return prev_ema + decay * tl.where(mask, input - prev_ema, 0)

@triton.jit
def input_standardization(input, mean, inv_std, mask: tl.constexpr):
    """
    Standardizes the input using mean and inverse standard deviation.

    Args:
        input: Input to standardize.
            The input must be of the same shape as the mask.
        mean: Mean of the input.
        inv_std: Inverse standard deviation of the input.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of the same shape as the input.

    Returns:
        Standardized input.
    """
    input = input.to(tl.float32)

    return (tl.where(mask, input - mean, 0) * inv_std)[:, None]

@triton.jit
def calc_l1_l2_loss(input1, input2, p, reduction: tl.constexpr):
    """
    Calculates the L1 or L2 loss between input1 and input2.

    Args:
        input1: First input.
            The first input must be of the same shape as the second input.
        input2: Second input.
            The second input must be of the same shape as the first input.
        p: Order of the norm to use for L1 or L2 loss.
        reduction: Type of reduction to apply to the output.
            Options are 'mean' and 'sum'.

    Returns:
        L1 or L2 loss between input1 and input2.
    """
    input1 = input1.to(tl.float32)
    input2 = input2.to(tl.float32)

    diff = input1 - input2
    loss = tl.sum(tl.pow(tl.abs(diff), p))

    if reduction == 'mean':
        loss /= input1.numel()

    return loss

@triton.jit
def calc_neg_log_likelihood_loss(input, target, logits, mask: tl.constexpr):
    """
    Calculates the negative log likelihood loss between input and target.

    Args:
        input: Input to calculate loss for.
            The input must be of the same shape as the mask.
        target: Target to calculate loss against.
            The target must be of the same shape as the input.
        logits: Logits of the input.
            The logits must be of the same shape as the input.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of the same shape as the input.

    Returns:
        Negative log likelihood loss between input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)
    logits = logits.to(tl.float32)

    log_probs = tl.where(mask, input - logits, 0)
    loss = tl.sum(tl.where(mask, target * log_probs, 0))

    return -loss

@triton.jit
def calc_cross_entropy_loss(input, target, logits, mask: tl.constexpr):
    """
    Calculates the cross entropy loss between input and target.

    Args:
        input: Input to calculate loss for.
            The input must be of the same shape as the mask.
        target: Target to calculate loss against.
            The target must be of the same shape as the input.
        logits: Logits of the input.
            The logits must be of the same shape as the input.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of the same shape as the input.

    Returns:
        Cross entropy loss between input and target.
    """
    input = input.to(tl.float32)
    target = target.to(tl.float32)
    logits = logits.to(tl.float32)

    log_probs = tl.where(mask, input - logits, 0)
    loss = tl.sum(tl.where(mask, target * log_probs, 0))

    return -loss

@triton.jit
def calc_l1_l2_norm(input, p, reduction: tl.constexpr):
    """
    Calculates the L1 or L2 norm of the input.

    Args:
        input: Input to calculate norm for.
        p: Order of the norm to use for L1 or L2 norm.
        reduction: Type of reduction to apply to the output.
            Options are 'mean' and 'sum'.

    Returns:
        L1 or L2 norm of the input.
    """
    input = input.to(tl.float32)

    norm = tl.sum(tl.pow(tl.abs(input), p))

    if reduction == 'mean':
        norm /= input.numel()

    return norm
<|endoftext|>
