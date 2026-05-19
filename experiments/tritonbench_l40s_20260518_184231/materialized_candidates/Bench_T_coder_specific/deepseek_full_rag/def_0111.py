Returns:
        Updated exponential moving average.
    """
    return (1 - momentum) * prev_ema + momentum * new_val

@triton.jit
def standardize(input, mean, inv_std, unbiased: tl.constexpr):
    """
    Standardizes the input using a mean and inverse standard deviation.

    Args:
        input: Input to standardize.
            The input must be of the same shape as mean and inv_std.
        mean: Mean used to standardize the input.
        inv_std: Inverse standard deviation used to standardize the input.
        unbiased: Flag for indicating if the unbiased standardization should be performed.

    Returns:
        Input standardized by the mean and inverse standard deviation.
    """
    input = input.to(tl.float32)

    output = (input - mean) * inv_std

    if unbiased:
        output = output * tl.rsqrt(1 - inv_std * inv_std)

    return output

@triton.jit
def calc_l1l2_norm(input, p: tl.constexpr, mask: tl.constexpr):
    """
    Calculates the L1 or L2 norm of the input.

    Args:
        input: Input whose L1 or L2 norm is calculated.
            The input must be of the same shape as the mask.
        p: Order of the norm.
            Options are 1 for L1 norm and 2 for L2 norm.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of the same shape as the input.

    Returns:
        L1 or L2 norm of the input.
    """
    input = input.to(tl.float32)

    return tl.sum(tl.where(mask, input, 0) ** p) ** (1 / p)

@triton.jit
def negative_log_likelihood(logits, labels, ignore_index: tl.constexpr):
    """
    Calculates the negative log likelihood of the logits for the labels.

    Args:
        logits: Logits from which to sample random labels.
            The logits must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        labels: Correct labels with integer values.
            The labels must be of shape [BLOCK_SIZE1].
        ignore_index: Index to ignore in the calculation of the negative log likelihood.

    Returns:
        Negative log likelihood of the logits for the labels.
    """
    logits = logits.to(tl.float32)

    logits_max = tl.max(logits, 1)[:, None]
    exp_logits = tl.exp(logits - logits_max)
    logits_sum = tl.sum(exp_logits, 1)[:, None]
    log_probs = logits - logits_max - tl.log(logits_sum)
    nll = -log_probs[tl.arange(0, logits.shape[0]), labels]
    nll = tl.where(labels != ignore_index, nll, 0)

    return nll

@triton.jit
def cross_entropy(logits, labels, ignore_index: tl.constexpr):
    """
    Calculates the cross entropy of the logits for the labels.

    Args:
        logits: Logits from which to sample random labels.
            The logits must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        labels: Correct labels with integer values.
            The labels must be of shape [BLOCK_SIZE1].
        ignore_index: Index to ignore in the calculation of the cross entropy.

    Returns:
        Cross entropy of the logits for the labels.
    """
    logits = logits.to(tl.float32)

    logits_max = tl.max(logits, 1)[:, None]
    exp_logits = tl.exp(logits - logits_max)
    logits_sum = tl.sum(exp_logits, 1)[:, None]
    logits = logits - logits_max - tl.log(logits_sum)
    ce = -logits[tl.arange(0, logits.shape[0]), labels]
    ce = tl.where(labels != ignore_index, ce, 0)

    return ce
